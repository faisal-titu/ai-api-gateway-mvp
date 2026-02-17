const API_URL = ""; // Served by same backend

// ==========================================
// CONFIGURATION 
const S3_BUCKET_NAME = "ai-image-searching";
const FOLDER_NAME = "unsplash";
const S3_REGION = "us-east-1";

// Construction of S3 URL
// Note: Your bucket must have public read access for these objects
// And CORS allowed for the frontend origin.
const S3_BASE_URL = `https://${S3_BUCKET_NAME}.s3.amazonaws.com/${FOLDER_NAME}`;
// ==========================================

const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const imageUpload = document.getElementById('imageUpload');
const gallery = document.getElementById('gallery');
const loading = document.getElementById('loading');
const resultsCount = document.getElementById('results-count');
const countSpan = document.getElementById('count');

// Search on Enter key
searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') performTextSearch();
});

// Search on Button Click
searchBtn.addEventListener('click', performTextSearch);

// Search on Image Upload
imageUpload.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    performImageSearch(file);
});

async function performTextSearch() {
    const query = searchInput.value.trim();
    if (!query) return;

    showLoading(true);
    gallery.innerHTML = '';
    resultsCount.classList.add('hidden');

    try {
        const response = await fetch(`${API_URL}/texts/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                num_images: 20
            })
        });

        if (!response.ok) throw new Error('Search failed');

        const data = await response.json();
        renderGallery(data.image_ids);

    } catch (error) {
        console.error(error);
        alert('Search failed. Please try again.');
    } finally {
        showLoading(false);
    }
}

async function performImageSearch(file) {
    showLoading(true);
    gallery.innerHTML = '';
    resultsCount.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('num_images', 20);

    try {
        const response = await fetch(`${API_URL}/images/search`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Search failed');

        const data = await response.json();
        renderGallery(data.image_ids);

    } catch (error) {
        console.error(error);
        alert('Image search failed.');
    } finally {
        showLoading(false);
    }
}

function renderGallery(imageIds) {
    if (!imageIds || imageIds.length === 0) {
        gallery.innerHTML = '<p style="text-align:center; width:100%;">No images found.</p>';
        return;
    }

    countSpan.textContent = imageIds.length;
    resultsCount.classList.remove('hidden');

    imageIds.forEach(id => {
        const item = document.createElement('div');
        item.className = 'gallery-item';

        const img = document.createElement('img');
        // Use S3 URL
        img.src = `${S3_BASE_URL}/${id}.jpg`;
        img.alt = `Image ${id}`;
        img.loading = "lazy";

        img.onload = () => img.classList.add('loaded');
        img.onerror = () => {
            // Only hide if load fails completely
            img.style.display = 'none';
        };

        item.appendChild(img);
        gallery.appendChild(item);
    });
}
function showLoading(isLoading) {
    if (isLoading) {
        loading.classList.remove('hidden');
    } else {
        loading.classList.add('hidden');
    }
}
