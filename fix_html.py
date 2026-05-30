import os

html_content = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nh?n di?n v?t th? trên bàn</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            text-align: center;
            background-color: #f4f7f6;
            margin: 0;
            padding: 20px;
        }
        h1 { color: #2c3e50; }
        p { color: #7f8c8d; }
        .upload-container {
            margin: 20px auto;
            padding: 20px;
            background: #fff;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            max-width: 500px;
        }
        input[type="file"] {
            margin-bottom: 15px;
        }
        button {
            padding: 10px 15px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            margin: 5px;
            transition: all 0.3s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .btn-group {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 10px;
            margin-top: 15px;
        }
        .result-container {
            margin-top: 30px;
        }
        img {
            border-radius: 8px;
            max-width: 100%;
            height: auto;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            display: none;
        }
        .loading {
            display: none;
            color: #e67e22;
            font-weight: bold;
            margin-top: 15px;
        }
    </style>
</head>
<body>
    <h1>Nh?n di?n v?t th? trên bàn làm vi?c</h1>
    <p>S? d?ng Python, FastAPI và YOLO</p>
    
    <div class="upload-container">
        <form id="uploadForm">
            <input type="file" id="imageInput" accept="image/*" required>
            
            <div style="font-weight: bold; margin-top: 15px; color: #34495e;">Nh?n di?n b?ng T? Khóa (YOLO-World v2):</div>
            <div class="btn-group">
                <button type="button" id="btnWorldX" style="background-color: #e67e22; width: 100%; font-size: 16px; padding: 12px;">Phân tích hình ?nh (B?n Extra Large)</button>
            </div>

            <div style="font-weight: bold; margin-top: 20px; color: #34495e;">Nh?n di?n t? File b?n Train:</div>
            <div class="btn-group">
                <button type="button" id="btnEnhance" style="background-color: #9b59b6; width: 100%;">Dùng Model Ðã Train (Best.pt)</button>
            </div>
        </form>
        <p class="loading" id="loadingText">Ðang phân tích AI, vui lòng ch?...</p>
        <p id="systemMessage" style="color: green; font-weight: bold; margin-top: 10px;"></p>
        <p id="pathInfo" style="font-size: 12px; color: #7f8c8d;"></p>
    </div>

    <div class="result-container">
        <h2>K?t qu?:</h2>
        <img id="resultImage" alt="K?t qu? nh?n di?n s? hi?n th? ? dây">
    </div>

    <script>
        async function submitForm(mode) {
            const fileInput = document.getElementById('imageInput');
            const file = fileInput.files[0];
            if (!file) {
                alert("Vui lòng ch?n ?nh tru?c!");
                return;
            }

            document.getElementById('loadingText').style.display = 'block';
            document.getElementById('resultImage').style.display = 'none';
            document.getElementById('systemMessage').innerText = "";
            document.getElementById('pathInfo').innerText = "";

            const formData = new FormData();
            formData.append('file', file);
            formData.append('mode', mode);

            try {
                const response = await fetch('/upload_image', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const data = await response.json();
                    
                    if (data.error) {
                        alert(data.error);
                    } else {
                        const imgElement = document.getElementById('resultImage');
                        imgElement.src = data.image_url;
                        imgElement.style.display = 'inline-block';
                        
                        document.getElementById('systemMessage').innerText = data.message;
                        document.getElementById('pathInfo').innerHTML = '?nh g?c: ' + data.raw_image_path + '<br>File nhãn: ' + data.label_file_path;
                    }
                } else {
                    alert('Có l?i x?y ra khi phân tích ?nh.');
                }
            } catch (error) {
                console.error(error);
                alert('Không th? k?t n?i d?n Server.');
            } finally {
                document.getElementById('loadingText').style.display = 'none';
            }
        }

        document.getElementById('btnWorldX').addEventListener('click', function() { submitForm('world_x'); });
        document.getElementById('btnEnhance').addEventListener('click', function() { submitForm('enhance'); });
    </script>
</body>
</html>"""

with open(r"F:\download\5s test\templates\index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("Ðã s?a HTML thành công!")
