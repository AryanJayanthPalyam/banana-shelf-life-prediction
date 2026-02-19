const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const previewBox = document.getElementById("imagePreviewBox");
const predictBtn = document.getElementById("predictBtn");

const loading = document.getElementById("loading");
const resultBox = document.getElementById("resultBox");

const varietyText = document.getElementById("varietyText");
const daysText = document.getElementById("daysText");

let uploadedFile = null;
/* Image Upload + Preview */
fileInput.addEventListener("change", function () {
    const file = fileInput.files[0];
    uploadedFile = file;

    if (file) {
        preview.src = URL.createObjectURL(file);
        previewBox.style.display = "block";

        // Show Predict button only after image loads
        preview.onload = function () {
            predictBtn.style.display = "block";
        };
    }
});
/* Send Image to FastAPI */
predictBtn.addEventListener("click", async function () {
    if (!uploadedFile) {
        alert("Please upload an image first");
        return;
    }
    predictBtn.style.display = "none";
    resultBox.style.display = "none";
    loading.style.display = "block";

    const formData = new FormData();
    formData.append("file", uploadedFile);

    try {
        const response = await fetch("http://localhost:8000/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        const days = data.predicted_days_until_death;

        varietyText.innerText = data.predicted_variety;

        if (days <= 0.5) {
            daysText.innerText = "Overripe / Rotten";
        } else {
            daysText.innerText = days + " days remaining";
        }

        resultBox.style.display = "block";

    } catch (err) {
        alert("Error connecting to FastAPI backend.");
    }

    loading.style.display = "none";
    predictBtn.style.display = "block";
});
