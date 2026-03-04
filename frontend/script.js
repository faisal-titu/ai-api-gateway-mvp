const API_URL = "";

// ==========================================
// CONFIGURATION
const S3_BUCKET_NAME = "ai-image-searching";
const FOLDER_NAME = "unsplash";
const S3_REGION = "us-east-1";

// Construction of S3 URL
const S3_BASE_URL = `https://${S3_BUCKET_NAME}.s3.amazonaws.com/${FOLDER_NAME}`;
// ==========================================

// Elements
const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const imageUpload = document.getElementById('imageUpload');
const kInput = document.getElementById('kInput');
const gallery = document.getElementById('gallery');
const loading = document.getElementById('loading');
const resultsBar = document.getElementById('results-bar');
const countSpan = document.getElementById('count');

// Preview Elements
const previewContainer = document.getElementById('image-preview-container');
const previewImg = document.getElementById('image-preview');
const clearPreviewBtn = document.getElementById('clear-preview');

// Lightbox Elements
const lightbox = document.getElementById('lightbox');
const lightboxImg = lightbox.querySelector('.lightbox-img');
const closeBtn = lightbox.querySelector('.close-lightbox');
const prevBtn = lightbox.querySelector('.prev-btn');
const nextBtn = lightbox.querySelector('.next-btn');

let currentImageIds = [];
let currentImageIndex = -1;
let currentUploadFile = null;

// Search on Enter key
searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') performSearch();
});

// Search on Button Click
searchBtn.addEventListener('click', performSearch);

// Handle Image Upload
imageUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    currentUploadFile = file;

    // Show Preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewContainer.classList.remove('hidden');
        previewContainer.classList.add('flex'); // Add flex for layout
    };
    reader.readAsDataURL(file);

    performImageSearch(file);
});

// Clear Preview
clearPreviewBtn.addEventListener('click', () => {
    currentUploadFile = null;
    previewImg.src = "";
    previewContainer.classList.add('hidden');
    previewContainer.classList.remove('flex');
    imageUpload.value = "";
});

async function performSearch() {
    if (currentUploadFile) {
        performImageSearch(currentUploadFile);
    } else {
        performTextSearch();
    }
}

async function performTextSearch() {
    const query = searchInput.value.trim();
    if (!query) return;

    showLoading(true);
    gallery.innerHTML = '';
    resultsBar.classList.add('hidden');

    const k = parseInt(kInput.value) || 20;

    try {
        const response = await fetch(`${API_URL}/texts/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query, num_images: k })
        });

        if (!response.ok) throw new Error('Search failed');

        const data = await response.json();
        currentImageIds = data.image_ids;
        renderGallery(currentImageIds);

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
    resultsBar.classList.add('hidden');

    const k = parseInt(kInput.value) || 20;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('num_images', k);

    try {
        const response = await fetch(`${API_URL}/images/search`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Search failed');

        const data = await response.json();
        currentImageIds = data.image_ids;
        renderGallery(currentImageIds);

    } catch (error) {
        console.error(error);
        alert('Image search failed.');
    } finally {
        showLoading(false);
    }
}

function renderGallery(imageIds) {
    if (!imageIds || imageIds.length === 0) {
        gallery.innerHTML = '<p class="col-span-full text-center text-gray-400 w-full py-10">No images found.</p>';
        return;
    }

    countSpan.textContent = imageIds.length;
    resultsBar.classList.remove('hidden');

    imageIds.forEach((id, index) => {
        const item = document.createElement('div');
        // Tailwind classes for gallery item
        item.className = 'relative overflow-hidden rounded-2xl cursor-zoom-in bg-white/5 aspect-square transition-all duration-300 hover:scale-105 hover:-translate-y-1 hover:shadow-2xl hover:z-10 group shadow-lg border border-white/5';

        const img = document.createElement('img');
        img.src = `${S3_BASE_URL}/${id}.jpg`;
        img.alt = `Image ${id}`;
        img.loading = "lazy";
        // Tailwind classes for image
        img.className = 'w-full h-full object-cover transition-opacity duration-500 opacity-0 group-hover:scale-110 transition-transform duration-700 ease-out';

        img.onload = () => img.classList.remove('opacity-0');
        img.onerror = () => { img.style.display = 'none'; };

        item.addEventListener('click', () => openLightbox(index));

        item.appendChild(img);
        gallery.appendChild(item);
    });
}

function showLoading(isLoading) {
    if (isLoading) {
        loading.classList.remove('hidden');
        loading.classList.add('flex');
    } else {
        loading.classList.add('hidden');
        loading.classList.remove('flex');
    }
}

// --- Lightbox Functions ---
function openLightbox(index) {
    if (index < 0 || index >= currentImageIds.length) return;
    currentImageIndex = index;
    const id = currentImageIds[index];

    lightboxImg.src = `${S3_BASE_URL}/${id}.jpg`;

    // Show lightbox (remove opacity-0 and pointer-events-none)
    lightbox.classList.remove('opacity-0', 'pointer-events-none');

    // Update button visibility
    prevBtn.classList.toggle('hidden', index === 0);
    nextBtn.classList.toggle('hidden', index === currentImageIds.length - 1);
}

function closeLightboxModal() {
    lightbox.classList.add('opacity-0', 'pointer-events-none');
}

function showNext() {
    if (currentImageIndex < currentImageIds.length - 1) {
        openLightbox(currentImageIndex + 1);
    }
}

function showPrev() {
    if (currentImageIndex > 0) {
        openLightbox(currentImageIndex - 1);
    }
}

// Lightbox Listeners
closeBtn.addEventListener('click', closeLightboxModal);
nextBtn.addEventListener('click', (e) => { e.stopPropagation(); showNext(); });
prevBtn.addEventListener('click', (e) => { e.stopPropagation(); showPrev(); });

lightbox.addEventListener('click', (e) => {
    // Close if clicking outside image
    if (e.target === lightbox) {
        closeLightboxModal();
    }
});

document.addEventListener('keydown', (e) => {
    if (!lightbox.classList.contains('pointer-events-none')) {
        if (e.key === 'Escape') closeLightboxModal();
        if (e.key === 'ArrowRight') showNext();
        if (e.key === 'ArrowLeft') showPrev();
    }
});
