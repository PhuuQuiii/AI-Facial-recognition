document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('imageUpload');
    const uploadButton = document.getElementById('uploadButton');

    const startWebcamButton = document.getElementById('startWebcamButton');
    const captureButton = document.getElementById('captureButton');
    const stopWebcamButton = document.getElementById('stopWebcamButton');
    const webcamFeed = document.getElementById('webcamFeed');

    const imagePreviewContainer = document.getElementById('imagePreviewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const imagePreviewCanvas = document.getElementById('imagePreviewCanvas');

    const resultsDiv = document.getElementById('results');
    const recognitionStatus = document.getElementById('recognitionStatus');
    const recognizedId = document.getElementById('recognizedId');
    const recognitionProbability = document.getElementById('recognitionProbability');
    const processingTime = document.getElementById('processingTime'); // This will show more detailed timing
    const facesDetectedCount = document.getElementById('facesDetectedCount');

    const loader = document.getElementById('loader');

    let stream;
    let currentImageForPreview = null;

    function showLoader() {
        loader.style.display = 'block';
    }

    function hideLoader() {
        loader.style.display = 'none';
    }

    function clearPreviousResults() {
        recognitionStatus.textContent = 'Trạng thái: Chưa có yêu cầu';
        recognizedId.textContent = 'MSSV: N/A';
        recognitionProbability.textContent = 'Xác suất: N/A';
        processingTime.textContent = 'Thời gian xử lý: N/A';
        facesDetectedCount.textContent = 'Số khuôn mặt: N/A';
        resultsDiv.style.display = 'none';
        const ctx = imagePreviewCanvas.getContext('2d');
        ctx.clearRect(0, 0, imagePreviewCanvas.width, imagePreviewCanvas.height);
        imagePreviewCanvas.style.display = 'none';
        imagePreview.style.display = 'block';
    }

    function drawImageWithBoxes(imageElementOrSrc, livenessResultData, recognitionApiData) {
        const img = new Image();
        img.onload = () => {
            const ctx = imagePreviewCanvas.getContext('2d');
            imagePreviewCanvas.width = img.naturalWidth || img.width;
            imagePreviewCanvas.height = img.naturalHeight || img.height;

            ctx.clearRect(0, 0, imagePreviewCanvas.width, imagePreviewCanvas.height);
            ctx.drawImage(img, 0, 0, imagePreviewCanvas.width, imagePreviewCanvas.height);

            if (livenessResultData && livenessResultData.face_coords) {
                const { x, y, w, h } = livenessResultData.face_coords;
                ctx.lineWidth = 3;
                if (livenessResultData.label === "Thật (Real)") {
                    ctx.strokeStyle = 'lime';
                } else if (livenessResultData.label === "Giả (Fake)") {
                    ctx.strokeStyle = 'red';
                } else {
                    ctx.strokeStyle = 'yellow';
                }
                ctx.strokeRect(x, y, w, h);
                ctx.fillStyle = ctx.strokeStyle;
                ctx.font = "16px Arial";
                ctx.fillText(livenessResultData.label || "Face", x, y > 20 ? y - 5 : y + h + 15);
            }

            if (recognitionApiData && recognitionApiData.recognitions && recognitionApiData.recognitions.length > 0) {
                recognitionApiData.recognitions.forEach(rec => {
                    if (rec.bbox) {
                        const [x1, y1, x2, y2] = rec.bbox;
                        ctx.strokeStyle = (rec.MSSV !== "Unknown" && rec.MSSV !== "ErrorInProcessing") ? 'cyan' : 'magenta';
                        ctx.lineWidth = 2;
                        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                        ctx.fillStyle = ctx.strokeStyle;
                        ctx.font = "14px Arial";
                        let text = `${rec.MSSV}`;
                        if (rec.probability) {
                            text += ` (${(rec.probability * 100).toFixed(1)}%)`;
                        }
                        ctx.fillText(text, x1, y1 > 10 ? y1 - 5 : y2 + 15);
                    }
                });
            }
            imagePreview.style.display = 'none';
            imagePreviewCanvas.style.display = 'block';
            imagePreviewContainer.style.display = 'block';
        };

        if (typeof imageElementOrSrc === 'string') {
            img.src = imageElementOrSrc;
        } else if (imageElementOrSrc && imageElementOrSrc.src) {
            img.src = imageElementOrSrc.src;
        } else if (imageElementOrSrc && imageElementOrSrc.tagName === 'VIDEO') {
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = imageElementOrSrc.videoWidth;
            tempCanvas.height = imageElementOrSrc.videoHeight;
            tempCanvas.getContext('2d').drawImage(imageElementOrSrc, 0, 0);
            img.src = tempCanvas.toDataURL('image/jpeg');
        }
        currentImageForPreview = img;
    }

    imageUpload.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                clearPreviousResults();
                drawImageWithBoxes(e.target.result, null, null);
                webcamFeed.style.display = 'none';
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                    startWebcamButton.disabled = false;
                    captureButton.disabled = true;
                    stopWebcamButton.disabled = true;
                }
            }
            reader.readAsDataURL(file);
        }
    });

    async function checkLiveness(imageBlob) {
        console.log("Bắt đầu kiểm tra liveness (gọi API /predict)...");
        const formData = new FormData();
        formData.append('file', imageBlob, 'liveness_check_image.jpg');

        let apiCallStartTime, apiCallEndTime, apiCallDurationSeconds;

        try {
            apiCallStartTime = performance.now(); // Start timing API call
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            // It's better to get the JSON before calculating end time if parsing is significant
            const data = await response.json();
            apiCallEndTime = performance.now(); // End timing API call
            apiCallDurationSeconds = ((apiCallEndTime - apiCallStartTime) / 1000).toFixed(3);

            if (!response.ok) {
                console.error("Lỗi API Liveness:", data); // data might contain error details from server
                return { isLive: false, data: null, error: data.error || `Lỗi máy chủ liveness: ${response.status}`, clientApiTime_s: apiCallDurationSeconds };
            }

            console.log("Kết quả API Liveness:", data);
            const isLive = data.label === "Thật (Real)";
            return { isLive, data, error: null, clientApiTime_s: apiCallDurationSeconds };

        } catch (error) {
            apiCallEndTime = performance.now(); // End timing even on catch
            apiCallDurationSeconds = apiCallStartTime ? ((apiCallEndTime - apiCallStartTime) / 1000).toFixed(3) : "N/A";
            console.error('Lỗi trong quá trình kiểm tra liveness:', error);
            return { isLive: false, data: null, error: `Lỗi client khi kiểm tra liveness: ${error.message}`, clientApiTime_s: apiCallDurationSeconds };
        }
    }

    function displayLivenessResult(livenessCheckResult, imageToDrawOn) {
        resultsDiv.style.display = 'block';
        if (!livenessCheckResult) return;

        const { isLive, data, error, clientApiTime_s } = livenessCheckResult;

        let livenessTimeDetails = `(API Call: ${clientApiTime_s || "N/A"}s`;
        if (data && data.processing_time) {
            livenessTimeDetails += `, Backend: ${data.processing_time}`;
        }
        livenessTimeDetails += ")";

        if (error) {
            recognitionStatus.textContent = `Trạng thái Liveness: Lỗi - ${error}`;
            processingTime.textContent = `Thời gian Liveness: ${livenessTimeDetails}`;
            drawImageWithBoxes(imageToDrawOn, { label: "Lỗi Liveness" }, null);
            return;
        }

        if (data) {
            recognitionStatus.textContent = `Trạng thái Liveness: ${data.label || "N/A"} (Điểm: ${data.score || "N/A"})`;
            processingTime.textContent = `Thời gian Liveness: ${livenessTimeDetails}`;
            facesDetectedCount.textContent = `Số khuôn mặt (Haar): ${data.num_faces !== undefined ? data.num_faces : 'N/A'}`;
            drawImageWithBoxes(imageToDrawOn, data, null);
        }
    }

    async function processImageAndRecognize(imageDataSource, type) {
        clearPreviousResults();
        let imageBlob = null;
        let imageSourceForPreviewDisplay = null;

        if (type === 'file') {
            if (!imageDataSource) {
                alert('Vui lòng chọn một file ảnh.');
                return;
            }
            imageBlob = imageDataSource;
            imageSourceForPreviewDisplay = currentImageForPreview.src;
        } else if (type === 'webcam') {
            if (!stream || !webcamFeed.srcObject || webcamFeed.paused || webcamFeed.ended) {
                alert('Webcam chưa được bật hoặc không có tín hiệu.');
                return;
            }
            const tempWebcamCanvas = document.createElement('canvas');
            tempWebcamCanvas.width = webcamFeed.videoWidth;
            tempWebcamCanvas.height = webcamFeed.videoHeight;
            tempWebcamCanvas.getContext('2d').drawImage(webcamFeed, 0, 0, tempWebcamCanvas.width, tempWebcamCanvas.height);
            
            imageSourceForPreviewDisplay = tempWebcamCanvas.toDataURL('image/jpeg');
            imageBlob = await new Promise(resolve => tempWebcamCanvas.toBlob(resolve, 'image/jpeg'));
            drawImageWithBoxes(imageSourceForPreviewDisplay, null, null);
        } else {
            return;
        }

        if (!imageBlob) {
            alert('Không thể lấy dữ liệu hình ảnh.');
            return;
        }

        showLoader();
        if (!currentImageForPreview && imageSourceForPreviewDisplay) {
             drawImageWithBoxes(imageSourceForPreviewDisplay, null, null);
        }

        const livenessCheckResult = await checkLiveness(imageBlob);
        displayLivenessResult(livenessCheckResult, currentImageForPreview || imageSourceForPreviewDisplay);

        if (!livenessCheckResult || !livenessCheckResult.isLive) {
            hideLoader();
            if (livenessCheckResult && livenessCheckResult.error) {
                 console.error("Liveness check failed or error:", livenessCheckResult.error);
            } else {
                 console.log("Liveness check: Khuôn mặt không phải thật hoặc có lỗi.");
                 recognitionStatus.textContent = `Trạng thái Liveness: ${livenessCheckResult?.data?.label || "Không thành công"}. Nhận diện bị dừng.`;
            }
            return;
        }

        resultsDiv.style.display = 'block';
        recognitionStatus.textContent = "Trạng thái Liveness: Thật. Đang tiến hành nhận diện...";
        console.log("Liveness PASSED. Proceeding to recognition...");

        const formDataRecognition = new FormData();
        formDataRecognition.append('image', imageBlob, type === 'file' ? imageBlob.name : 'webcam_capture.jpg');
        
        let recognitionApiCallStartTime, recognitionApiCallEndTime, recognitionApiCallDurationSeconds;

        try {
            recognitionApiCallStartTime = performance.now(); // Start timing recognition API call
            const responseRecognition = await fetch('http://127.0.0.1:5001/recognize', {
                method: 'POST',
                body: formDataRecognition
            });
            const recognitionApiData = await responseRecognition.json();
            recognitionApiCallEndTime = performance.now(); // End timing recognition API call
            recognitionApiCallDurationSeconds = ((recognitionApiCallEndTime - recognitionApiCallStartTime) / 1000).toFixed(3);


            if (!responseRecognition.ok) {
                console.error("Lỗi API Recognition:", recognitionApiData);
                throw new Error(recognitionApiData.message || `Lỗi HTTP nhận diện: ${responseRecognition.status}`);
            }
            
            console.log("Kết quả API Recognition:", recognitionApiData);
            // Add client-side measured time to the data to pass to display function
            recognitionApiData.clientApiTime_s = recognitionApiCallDurationSeconds;
            displayRecognitionResults(recognitionApiData, livenessCheckResult.data, livenessCheckResult.clientApiTime_s);

        } catch (error) {
            recognitionApiCallEndTime = performance.now(); // End timing even on catch
            recognitionApiCallDurationSeconds = recognitionApiCallStartTime ? ((recognitionApiCallEndTime - recognitionApiCallStartTime) / 1000).toFixed(3) : "N/A";
            console.error('Lỗi xử lý nhận diện:', error);
            recognitionStatus.textContent = `Trạng thái Nhận diện: Lỗi - ${error.message}`;
            processingTime.textContent = `Thời gian Liveness: (API Call: ${livenessCheckResult.clientApiTime_s || "N/A"}s, Backend: ${livenessCheckResult.data?.processing_time || "N/A"}) | Nhận diện: (API Call: ${recognitionApiCallDurationSeconds}s, Lỗi)`;
            drawImageWithBoxes(currentImageForPreview || imageSourceForPreviewDisplay, livenessCheckResult.data, {
                message: `Lỗi nhận diện: ${error.message}`,
                recognitions: []
            });
        } finally {
            hideLoader();
        }
    }

    uploadButton.addEventListener('click', () => {
        if (!imageUpload.files[0]) {
            alert("Vui lòng chọn file ảnh trước.");
            return;
        }
        processImageAndRecognize(imageUpload.files[0], 'file');
    });

    captureButton.addEventListener('click', () => {
        processImageAndRecognize(webcamFeed, 'webcam');
    });

    startWebcamButton.addEventListener('click', async () => {
        clearPreviousResults();
        try {
            if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 } } });
                webcamFeed.srcObject = stream;
                webcamFeed.style.display = 'block';
                imagePreviewContainer.style.display = 'none';
                startWebcamButton.disabled = true;
                captureButton.disabled = false;
                stopWebcamButton.disabled = false;
            } else {
                alert('Trình duyệt của bạn không hỗ trợ API webcam.');
            }
        } catch (error) {
            console.error('Lỗi khi mở webcam:', error);
            alert('Không thể truy cập webcam. Vui lòng kiểm tra quyền truy cập hoặc webcam khác đang sử dụng.');
        }
    });

    stopWebcamButton.addEventListener('click', () => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            webcamFeed.srcObject = null;
            webcamFeed.style.display = 'none';
            startWebcamButton.disabled = false;
            captureButton.disabled = true;
            stopWebcamButton.disabled = true;
            clearPreviousResults();
        }
    });

    // Modified to accept livenessClientApiTime_s
    function displayRecognitionResults(recognitionData, livenessApiData, livenessClientApiTime_s) {
        resultsDiv.style.display = 'block';

        if (!recognitionData) {
            recognitionStatus.textContent = `Trạng thái Nhận diện: Lỗi - Không có dữ liệu trả về.`;
            return;
        }
        
        recognitionStatus.textContent = `Trạng thái Nhận diện: ${recognitionData.message || (recognitionData.success ? 'Thành công' : 'Thất bại hoặc không rõ')}`;
        facesDetectedCount.textContent = `Số khuôn mặt (Nhận diện): ${recognitionData.faces_detected_count !== undefined ? recognitionData.faces_detected_count : (recognitionData.recognitions ? recognitionData.recognitions.length : 'N/A')}`;

        const firstRecognized = recognitionData.recognitions ? recognitionData.recognitions.find(r => r.MSSV !== "Unknown" && r.MSSV !== "ErrorInProcessing") : null;

        if (firstRecognized) {
            recognizedId.textContent = `MSSV: ${firstRecognized.MSSV}`;
            recognitionProbability.textContent = `Xác suất: ${(firstRecognized.probability * 100).toFixed(1)}%`;
        } else if (recognitionData.recognitions?.length > 0 && recognitionData.recognitions[0].MSSV === "Unknown") {
            recognizedId.textContent = 'MSSV: Không xác định (Unknown)';
            recognitionProbability.textContent = `Xác suất (cao nhất cho Unknown): ${(recognitionData.recognitions[0].probability * 100).toFixed(1)}%`;
        } else if (recognitionData.recognitions?.length > 0 && recognitionData.recognitions[0].MSSV === "ErrorInProcessing") {
             recognizedId.textContent = 'MSSV: Lỗi xử lý khuôn mặt khi nhận diện';
             recognitionProbability.textContent = 'Xác suất: N/A';
        } else if (!recognitionData.recognitions || recognitionData.recognitions.length === 0) {
            recognizedId.textContent = 'MSSV: Không tìm thấy/nhận diện được khuôn mặt nào.';
            recognitionProbability.textContent = 'Xác suất: N/A';
        } else {
            recognizedId.textContent = 'MSSV: N/A';
            recognitionProbability.textContent = 'Xác suất: N/A';
        }
        
        // Update processing time display
        const livenessBackendTime = livenessApiData?.processing_time || "N/A";
        const recognitionClientTime = recognitionData.clientApiTime_s || "N/A"; // From recognitionData itself
        const recognitionBackendTime = recognitionData.processing_time_ms ? (recognitionData.processing_time_ms / 1000).toFixed(2) + 's' : 'N/A';

        processingTime.textContent = `Liveness (API: ${livenessClientApiTime_s}s, Backend: ${livenessBackendTime}) | Nhận diện (API: ${recognitionClientTime}s, Backend: ${recognitionBackendTime})`;

        drawImageWithBoxes(currentImageForPreview || imagePreview.src, livenessApiData, recognitionData);
    }
});