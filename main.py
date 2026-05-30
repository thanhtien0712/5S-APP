import cv2
import numpy as np
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import io
import os
import time
import aiofiles
import asyncio
import google.generativeai as genai
from PIL import Image

from pydantic import BaseModel
from typing import List, Dict, Any

class RawObject(BaseModel):
    name: str
    box: List[int]
    conf: float

class Step2Request(BaseModel):
    safe_filename: str
    raw_objects: List[RawObject]

class DetectedObjectStatus(BaseModel):
    name: str
    status: str

class Step3Request(BaseModel):
    detected_objects: List[DetectedObjectStatus]

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

DATASET_IMG_DIR = "dataset/images/train"
os.makedirs(DATASET_IMG_DIR, exist_ok=True)
os.makedirs("static/results", exist_ok=True)

# CẤU HÌNH GEMINI API
gemini_available = False
try:
    # Bảo mật: Đọc từ biến môi trường trước (cho Vercel/Render), nếu không có mới đọc file local
    API_KEY = os.environ.get("GEMINI_API_KEY")
    if not API_KEY and os.path.exists('api_key.txt'):
        with open('api_key.txt', 'r', encoding='utf-8') as f:
            API_KEY = f.read().strip()
            
    if API_KEY:
        genai.configure(api_key=API_KEY)
        gemini_model = genai.GenerativeModel('models/gemini-2.5-flash')
        gemini_available = True
        print("Đã khởi tạo thành công Gemini 2.5 Flash (Auto-Corrector)!")
    else:
        print("CẢNH BÁO: Chưa cấu hình GEMINI_API_KEY!")
except Exception as e:
    print(f"Không thể khởi tạo Gemini: {e}")

print("Đang nạp các mô hình AI cùng lúc...")
model_world_x = None
custom_classes_world = []

try:
    from ultralytics import YOLO
    
    default_classes = ["desk", "table", "cubicle partition", "wall", "partition", "laptop", "monitor", "computer mouse", "keyboard", 
                       "smartphone", "mug", "box", "tissue box", "spray bottle", "desk phone","glasses", "notebook", "pen", "hand fan", "headphones", "backpack", "water bottle", "calculator", "sticky notes"]
    custom_classes_world = default_classes
    
    print("-> YOLO-World sẽ tìm các từ khóa:", custom_classes_world)
    model_world_x = YOLO("yolov8x-worldv2.pt")
    model_world_x.set_classes(custom_classes_world)
    
    print("Sẵn sàng! Tất cả mô hình đã được nạp!")
except Exception as e:
    print(f"LỖI khởi tạo AI: {e}")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

# Hàm nhờ Gemini kiểm tra GỘP tất cả món đồ 1 lần trên ảnh toàn cảnh
async def ask_gemini_to_correct_batch(img_full, objects_list):
    if not gemini_available or not objects_list: 
        return {i: obj['name'] for i, obj in enumerate(objects_list)}
        
    try:
        from PIL import ImageDraw, ImageFont
        
        # Lấy ảnh mặt bàn để Gemini có bối cảnh toàn diện
        img_pil = Image.fromarray(cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB))
        
        # Giữ kích thước vừa phải (800px) để tăng tốc độ upload và xử lý mà không quá mờ
        max_dim = 900
        if img_pil.width > max_dim or img_pil.height > max_dim:
            img_pil.thumbnail((max_dim, max_dim))
            
        draw = ImageDraw.Draw(img_pil)
        
        prompt = "Đây là khu vực bàn làm việc. Các vật thể đã được AI khoanh vùng và đánh số ID (màu trắng nền đỏ) kèm khung xanh lá.\n"
        prompt += "Dưới đây là danh sách tên dự đoán và tọa độ [x1, y1, x2, y2]:\n"
        
        for i, obj in enumerate(objects_list):
            name = obj['name']
            x1, y1, x2, y2 = obj['box']
            
            # Tính tọa độ theo tỷ lệ đã thu nhỏ
            ratio_x = img_pil.width / img_full.shape[1]
            ratio_y = img_pil.height / img_full.shape[0]
            rx1, ry1 = int(x1 * ratio_x), int(y1 * ratio_y)
            rx2, ry2 = int(x2 * ratio_x), int(y2 * ratio_y)
            
            # Vẽ lên ảnh cho Gemini nhìn
            draw.rectangle([rx1, ry1, rx2, ry2], outline="green", width=3)
            draw.rectangle([rx1, max(0, ry1-25), rx1+40, ry1], fill="red")
            draw.text((rx1+5, max(0, ry1-20)), str(i), fill="white")
            
            prompt += f"ID {i}: Predicted '{name}', Coords: [{x1}, {y1}, {x2}, {y2}]\n"
            
        prompt += """
        Act as a data cleaning and evaluation expert.
        MANDATORY RULES (CRITICAL: REMOVE DUPLICATE/OVERLAPPING BOXES):
        1. REMOVE INNER DETAILS (MERGE): 
           - If a "keyboard", "monitor" or "screen" is INSIDE the coordinates of a "laptop", mark it as "delete". Keep only the overall "laptop" ID.
           - If a "straw", "lid" is inside a "water cup/mug", "delete" those extra details, keeping only 1 ID for the cup/mug.
        2. REMOVE OVERLAPPING BOXES ON THE SAME OBJECT:
           - If 2 IDs bound the same cup (or highly overlapping), keep the one with the best bounding box, mark the other as "delete".
        3. CORRECT WRONG NAMES & CHECK FOR SHARP OBJECTS: 
           - If an object is misclassified (e.g., spray bottle called water bottle), correct the name (e.g., spray bottle).
           - EXTREMELY IMPORTANT: Carefully evaluate "pen" objects. If it looks like a box cutter, fruit knife, knife, or scissors, rename it immediately to "knife" or "scissors".
        4. IGNORE NOISE: Mark empty spaces, shadows, or body parts (hands, arms) as "delete".
        
        TO SPEED UP PROCESSING, ONLY RETURN IDs THAT NEED RENAMING OR DELETING. DO NOT list correct IDs.
        Return EXACTLY in this format (1 per line): ID -> New_Name_or_delete
        No explanations.
        """
        
        print(f"[Gemini] Sending overview image for deduplication of {len(objects_list)} objects...")
        response = await asyncio.to_thread(gemini_model.generate_content, [prompt, img_pil])
        correction_text = response.text.lower().strip()
        
        corrected_names = {}
        for line in correction_text.split('\n'):
            if "->" in line:
                parts = line.split("->")
                try:
                    idx = int(parts[0].replace('id', '').strip())
                    new_name = parts[1].strip()
                    new_name = ''.join(e for e in new_name if e.isalnum() or e.isspace())
                    corrected_names[idx] = new_name
                    print(f"   [Gemini] Processed ID {idx}: {objects_list[idx]['name']} -> {new_name}")
                except:
                    pass
                    
        return corrected_names
    except Exception as e:
        print(f"[Gemini] Lỗi phân tích toàn cảnh: {e}")
        return {}

async def generate_5s_report(detected_objects):
    if not gemini_available or not detected_objects:
        return "<p>Không thể tạo báo cáo 5S do thiếu dữ liệu hoặc AI chưa sẵn sàng.</p>"
    
    # Tính điểm trực tiếp bằng Python để đồng bộ 100% với khung đỏ
    good_items = [obj.name for obj in detected_objects if obj.status == 'good']
    bad_items = [obj for obj in detected_objects if obj.status != 'good']
    
    score = 100
    for obj in bad_items:
        if obj.status == 'over_quota':
            score -= 5
        elif obj.status == 'unauthorized':
            score -= 10
            
    score = max(0, score)

    prompt = f"""
The AI system (Python) has scored and categorized the items on the desk as follows:
- COMPLIANT items (Green Box): {", ".join(set(good_items)) if good_items else "None"}
- VIOLATING items (Red Box, points deducted):
"""
    if not bad_items:
        prompt += "  + No violations found!\n"
    for obj in bad_items:
        reason = "Exceeds allowed quantity (-5 pts)" if obj.status == 'over_quota' else "Not in standard (-10 pts)"
        prompt += f"  + {obj.name}: {reason}\n"

    prompt += f"""
TOTAL SCORE: {score}/100

Write a 5S audit report based on the EXACT DATA PROVIDED ABOVE.
MANDATORY REQUIREMENTS:
1. NO greetings, NO conclusions. Use the exact score {score}/100 calculated by the system. Write entirely in English.
2. Present strictly in the following HTML structure, divided into 2 sections (COMPLIANT and ACTION REQUIRED), DO NOT use markdown ```html:

<h2 class="score-heading">5S SCORE: {score}/100</h2>
<div class="box-good">
    <h3>✅ COMPLIANT ITEMS</h3>
    <ul>
        <li>(Briefly list the COMPLIANT items above)</li>
    </ul>
</div>
<div class="box-bad">
    <h3>⚠️ ACTION REQUIRED</h3>
    <ul>
        <li>(List the VIOLATING items and the EXACT deduction reason provided by the system)</li>
    </ul>
</div>
    """
    
    try:
        print("[Gemini] Generating 5S audit report...")
        # Sử dụng temperature thấp để output ổn định, tránh ảo giác
        generation_config = genai.types.GenerationConfig(temperature=0.2)
        response = await asyncio.to_thread(gemini_model.generate_content, prompt, generation_config=generation_config)
        return response.text.replace('```html', '').replace('```', '').strip()
    except Exception as e:
        print(f"[Gemini] Lỗi tạo báo cáo 5S: {e}")
        return f"<p>Lỗi tạo báo cáo 5S: {str(e)}</p>"

@app.post("/api/step1_yolo")
async def step1_yolo(file: UploadFile = File(...)):
    if not model_world_x:
        return JSONResponse({"error": "Lỗi khởi tạo AI. Hãy xem log terminal."}, status_code=500)

    contents = await file.read()
    img_filename = file.filename
    
    file_id = str(int(time.time()))
    safe_filename = f"{file_id}_{img_filename.replace(' ', '_')}"
    img_save_path = os.path.join(DATASET_IMG_DIR, safe_filename)
    
    async with aiofiles.open(img_save_path, 'wb') as out_file:
        await out_file.write(contents)
    
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    height, width, _ = frame.shape
    
    active_model = model_world_x
    active_classes = custom_classes_world
    conf_threshold = 0.08

    results_desk = active_model.predict(frame, conf=0.08, iou=0.4) 
    
    max_desk_area = 0
    desk_bbox = None
    y_min_anchor = height
    y_min_partition = height
    has_partition = False
    
    for result in results_desk:
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id < len(active_classes):
                class_name = active_classes[class_id]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                area = (x2 - x1) * (y2 - y1)
                
                if class_name in ["desk", "table"]:
                    if area > max_desk_area:
                        max_desk_area = area
                        desk_bbox = (x1, y1, x2, y2)
                        
                if class_name in ["cubicle partition", "wall", "partition"]:
                    has_partition = True
                    if y1 < y_min_partition:
                        y_min_partition = y1
                        
                if class_name in ["laptop", "monitor"]:
                    if y1 < y_min_anchor:
                        y_min_anchor = y1

    if not desk_bbox:
        margin_x = int(width * 0.10)
        desk_bbox = (margin_x, int(height * 0.10), width - margin_x, height)
        
    dx1, dy1, dx2, dy2 = desk_bbox
    partition_margin = int(height * 0.10)
    
    if has_partition:
        dy1 = min(dy1, y_min_partition)
    else:
        if y_min_anchor < height and y_min_anchor > 0:
            dy1 = max(0, y_min_anchor - partition_margin)
        else:
            dy1 = max(0, dy1 - partition_margin)
        
    dx1 = max(0, dx1 - 20)
    dx2 = min(width, dx2 + 20)
    dy2 = min(height, dy2 + 20)
    
    desk_crop = frame[dy1:dy2, dx1:dx2]
    
    results_objects = active_model.predict(desk_crop, conf=conf_threshold, iou=0.15)
    raw_objects = []
    
    for result in results_objects:
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id >= len(active_classes):
                continue
                
            original_class_name = active_classes[class_id]
            if original_class_name in ["desk", "table", "cubicle partition", "wall", "partition"]:
                continue
                
            conf = float(box.conf[0])
            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = bx1 + dx1, by1 + dy1, bx2 + dx1, by2 + dy1
            
            raw_objects.append({
                "name": original_class_name,
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "conf": float(conf)
            })
            
            # Vẽ nháp khung màu xanh lơ mờ cho bước 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
            cv2.putText(frame, original_class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

    result_img_path = f"static/results/{safe_filename}_step1.jpg"
    cv2.imwrite(result_img_path, frame)
    
    return JSONResponse({
        "success": True,
        "safe_filename": safe_filename,
        "raw_objects": raw_objects,
        "image_url": f"/{result_img_path}",
        "message": "Spatial recognition complete."
    })

@app.post("/api/step2_clean")
async def step2_clean(req: Step2Request):
    safe_filename = req.safe_filename
    raw_objects = [obj.dict() for obj in req.raw_objects]
    
    img_save_path = os.path.join(DATASET_IMG_DIR, safe_filename)
    frame = cv2.imread(img_save_path)
    
    if frame is None:
        return JSONResponse({"error": "Không tìm thấy ảnh gốc."}, status_code=400)
    
    detected_objects = []
    
    if raw_objects:
        if gemini_available:
            corrected_names_dict = await ask_gemini_to_correct_batch(frame, raw_objects)
        else:
            corrected_names_dict = {}

        object_counter = 1
        seen_classes_in_image = set()
        
        # Đếm tần suất các món đồ ĐÃ CHUẨN HÓA để check vi phạm số lượng
        from collections import Counter
        item_frequencies = Counter()
        final_objects_info = []
        
        for i, obj in enumerate(raw_objects):
            final_name = corrected_names_dict.get(i, obj["name"])
            if final_name in ['delete', 'x', 'remove', 'drop', 'none', '']:
                continue
            
            item_frequencies[final_name] += 1
            final_objects_info.append({
                "raw_index": i,
                "box": obj["box"],
                "conf": obj["conf"],
                "final_name": final_name,
                "occurrence_index": item_frequencies[final_name] # Đây là món thứ mấy của loại này
            })

        # Định nghĩa quota cho các mặt hàng tốt dựa theo system prompt
        quota_limits = {
            'laptop': 2, 'monitor': 2, 'keyboard': 1, 'computer mouse': 1,
            'water bottle': 2, 'mug': 2, 'desk phone': 1, 'tissue box': 1, 'pen': 5, 'notebook': 2, 'smartphone': 1, 'calculator': 1, 'backpack': 1
        }

        for info in final_objects_info:
            x1, y1, x2, y2 = info["box"]
            conf = info["conf"]
            final_name = info["final_name"]
            occ_index = info["occurrence_index"]
            
            display_label = f"[{object_counter}] {final_name} {conf:.2f}"
            
            # Kiểm tra xem có vi phạm hay không (Vượt Quota hoặc Không có trong danh sách chuẩn)
            is_unauthorized = final_name not in quota_limits
            is_over_quota = not is_unauthorized and occ_index > quota_limits[final_name]
            
            if is_unauthorized:
                status = "unauthorized"
            elif is_over_quota:
                status = "over_quota"
            else:
                status = "good"

            detected_objects.append({
                "name": final_name,
                "status": status
            })
            
            if is_unauthorized or is_over_quota:
                box_color = (0, 0, 255)      # Đỏ (BGR)
                label_bg_color = (0, 0, 150)
            else:
                box_color = (0, 255, 0)      # Xanh lá cây (BGR)
                label_bg_color = (0, 150, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            (w, h), _ = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), label_bg_color, -1) 
            cv2.putText(frame, display_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            object_counter += 1

    result_img_path = f"static/results/{safe_filename}_step2.jpg"
    cv2.imwrite(result_img_path, frame)
    
    return JSONResponse({
        "success": True,
        "detected_objects": detected_objects,
        "image_url": f"/{result_img_path}",
        "message": "Data normalization successful."
    })

@app.post("/api/step3_report")
async def step3_report(req: Step3Request):
    report_5s = await generate_5s_report(req.detected_objects) if req.detected_objects else "<p>No objects detected on the desk for evaluation.</p>"
    return JSONResponse({
        "success": True,
        "report_5s": report_5s,
        "message": "Report generation complete."
    })
