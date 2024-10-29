document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault();
    const fileInput = document.getElementById('imageInput');
    const file = fileInput.files[0];

    const reader = new FileReader();
    reader.onloadend = function() {
        const base64Image = reader.result.split(',')[1]; // Lấy phần base64
        fetch('/attendance', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: new URLSearchParams({
                'image': base64Image
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.name) {
                document.getElementById('result').innerText = `Chào mừng, ${data.name}!`;
            } else {
                document.getElementById('result').innerText = 'Không nhận diện được khuôn mặt.';
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    };
    reader.readAsDataURL(file);
});

document.getElementById('checkAttendanceButton').addEventListener('click', function() {
    fetch('/check_attendance')
    .then(response => response.json())
    .then(data => {
        const list = document.getElementById('attendanceList');
        list.innerHTML = ''; // Xóa danh sách cũ
        if (data.attended_students.length > 0) {
            data.attended_students.forEach(student => {
                const item = document.createElement('div');
                item.innerText = student;
                list.appendChild(item);
            });
        } else {
            list.innerText = 'Không có sinh viên nào đã điểm danh.';
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
});