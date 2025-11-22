// Get API URL from environment variable or use default
const DEFAULT_API_URL = 'https://fpqr9yfck4x72w-8003.proxy.runpod.net';

// Wait for marked.js to load and configure it
function configureMarked() {
    if (typeof marked !== 'undefined') {
        marked.setOptions({
            breaks: true,  // Convert \n to <br>
            gfm: true,     // GitHub Flavored Markdown
            headerIds: false,
            mangle: false
        });
        return true;
    }
    return false;
}

// Function to render markdown/HTML content
function renderMarkdown(text) {
    if (!text) return '';
    
    // Check if marked is available
    if (typeof marked !== 'undefined' && typeof marked.parse === 'function') {
        try {
            // marked.js preserves HTML by default, so we can safely parse
            // This will convert markdown to HTML while preserving existing HTML tags
            const result = marked.parse(text);
            return result;
        } catch (e) {
            console.error('Markdown parsing error:', e);
            // Fallback: if it looks like HTML, return as-is
            if (/<[a-z][\s\S]*>/i.test(text)) {
                return text;
            }
            // Otherwise escape HTML entities
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    } else {
        console.warn('marked.js not loaded yet, using fallback rendering');
        // Fallback: if it contains HTML tags, return as-is
        if (/<[a-z][\s\S]*>/i.test(text)) {
            return text;
        }
        // Otherwise escape and return as plain text
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Language options for translation
const LANGUAGE_OPTIONS = [
    { code: 'ar_AR', name: 'Arabic' },
    { code: 'cs_CZ', name: 'Czech' },
    { code: 'de_DE', name: 'German' },
    { code: 'en_XX', name: 'English' },
    { code: 'es_XX', name: 'Spanish' },
    { code: 'et_EE', name: 'Estonian' },
    { code: 'fi_FI', name: 'Finnish' },
    { code: 'fr_XX', name: 'French' },
    { code: 'gu_IN', name: 'Gujarati' },
    { code: 'hi_IN', name: 'Hindi' },
    { code: 'it_IT', name: 'Italian' },
    { code: 'ja_XX', name: 'Japanese' },
    { code: 'kk_KZ', name: 'Kazakh' },
    { code: 'ko_KR', name: 'Korean' },
    { code: 'lt_LT', name: 'Lithuanian' },
    { code: 'lv_LV', name: 'Latvian' },
    { code: 'my_MM', name: 'Burmese' },
    { code: 'ne_NP', name: 'Nepali' },
    { code: 'nl_XX', name: 'Dutch' },
    { code: 'ro_RO', name: 'Romanian' },
    { code: 'ru_RU', name: 'Russian' },
    { code: 'si_LK', name: 'Sinhala' },
    { code: 'tr_TR', name: 'Turkish' },
    { code: 'vi_VN', name: 'Vietnamese' },
    { code: 'zh_CN', name: 'Chinese' },
    { code: 'af_ZA', name: 'Afrikaans' },
    { code: 'az_AZ', name: 'Azerbaijani' },
    { code: 'bn_IN', name: 'Bengali' },
    { code: 'fa_IR', name: 'Persian' },
    { code: 'he_IL', name: 'Hebrew' },
    { code: 'hr_HR', name: 'Croatian' },
    { code: 'id_ID', name: 'Indonesian' },
    { code: 'ka_GE', name: 'Georgian' },
    { code: 'km_KH', name: 'Khmer' },
    { code: 'mk_MK', name: 'Macedonian' },
    { code: 'ml_IN', name: 'Malayalam' },
    { code: 'mn_MN', name: 'Mongolian' },
    { code: 'mr_IN', name: 'Marathi' },
    { code: 'pl_PL', name: 'Polish' },
    { code: 'ps_AF', name: 'Pashto' },
    { code: 'pt_XX', name: 'Portuguese' },
    { code: 'sv_SE', name: 'Swedish' },
    { code: 'sw_KE', name: 'Swahili' },
    { code: 'ta_IN', name: 'Tamil' },
    { code: 'te_IN', name: 'Telugu' },
    { code: 'th_TH', name: 'Thai' },
    { code: 'tl_XX', name: 'Tagalog' },
    { code: 'uk_UA', name: 'Ukrainian' },
    { code: 'ur_PK', name: 'Urdu' },
    { code: 'xh_ZA', name: 'Xhosa' },
    { code: 'gl_ES', name: 'Galician' },
    { code: 'sl_SI', name: 'Slovene' }
];

// Function to populate language dropdowns
function populateLanguageDropdowns() {
    const srcLangSelect = document.getElementById('srcLang');
    const targetLangSelect = document.getElementById('targetLang');
    
    if (srcLangSelect && targetLangSelect) {
        // Sort languages by name for easier selection
        const sortedLanguages = [...LANGUAGE_OPTIONS].sort((a, b) => a.name.localeCompare(b.name));
        
        sortedLanguages.forEach(lang => {
            const srcOption = document.createElement('option');
            srcOption.value = lang.code;
            srcOption.textContent = `${lang.name} (${lang.code})`;
            srcLangSelect.appendChild(srcOption);
            
            const targetOption = document.createElement('option');
            targetOption.value = lang.code;
            targetOption.textContent = `${lang.name} (${lang.code})`;
            targetLangSelect.appendChild(targetOption);
        });
        
        // Set default values (English as source, Arabic as target)
        srcLangSelect.value = 'en_XX';
        targetLangSelect.value = 'ar_AR';
    }
}

// Initialize API URL from localStorage or use default
let apiUrl = localStorage.getItem('apiUrl') || DEFAULT_API_URL;

// Initialize when ready
function initApp() {
    // Configure marked.js
    if (typeof marked !== 'undefined') {
        configureMarked();
    }
    
    // Set API URL input
    const apiUrlInput = document.getElementById('apiUrl');
    if (apiUrlInput) {
        apiUrlInput.value = apiUrl;
    }
    
    // Populate language dropdowns
    populateLanguageDropdowns();
}

// Wait for DOM and marked.js to be ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    // DOM already loaded, but wait a bit for marked.js
    setTimeout(initApp, 50);
}

// Save API URL
function saveApiUrl() {
    const input = document.getElementById('apiUrl').value.trim();
    if (input) {
        apiUrl = input.replace(/\/$/, ''); // Remove trailing slash
        localStorage.setItem('apiUrl', apiUrl);
        showNotification('API URL saved successfully', 'success');
    } else {
        showNotification('Please enter a valid API URL', 'error');
    }
}

// Simple notification system
function showNotification(message, type = 'info') {
    // Remove existing notification if any
    const existing = document.querySelector('.notification');
    if (existing) {
        existing.remove();
    }
    
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
        <span>${message}</span>
    `;
    
    document.body.appendChild(notification);
    
    // Trigger animation
    setTimeout(() => notification.classList.add('show'), 10);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Tab switching
function switchTab(tab, event) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(`${tab}-tab`).classList.add('active');
    if (event && event.target) {
        event.target.classList.add('active');
    } else {
        // Fallback: find button by tab name
        document.querySelectorAll('.tab-button').forEach(button => {
            if (button.textContent.toLowerCase().includes(tab)) {
                button.classList.add('active');
            }
        });
    }
}

// File handling
let selectedFile = null;

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        selectedFile = file;
        document.getElementById('fileName').textContent = file.name;
        document.getElementById('fileInfo').style.display = 'block';
        document.getElementById('ocrResults').style.display = 'none';
        document.getElementById('ocrError').style.display = 'none';
    }
}

function clearFile() {
    selectedFile = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('fileInfo').style.display = 'none';
    document.getElementById('ocrResults').style.display = 'none';
    document.getElementById('ocrError').style.display = 'none';
}

// Drag and drop
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');

uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const file = e.dataTransfer.files[0];
    if (file) {
        fileInput.files = e.dataTransfer.files;
        handleFileSelect({ target: { files: [file] } });
    }
});

// OCR Processing
async function processOCR() {
    if (!selectedFile) {
        showNotification('Please select a file first', 'error');
        return;
    }
    
    const loadingDiv = document.getElementById('ocrLoading');
    const resultsDiv = document.getElementById('ocrResults');
    const errorDiv = document.getElementById('ocrError');
    
    // Show loading, hide results and errors
    loadingDiv.style.display = 'block';
    resultsDiv.style.display = 'none';
    errorDiv.style.display = 'none';
    
    try {
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        const response = await fetch(`${apiUrl}/ocr/infer`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Hide loading, show results
        loadingDiv.style.display = 'none';
        resultsDiv.style.display = 'block';
        
        // Scroll to results
        resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        
        // Display metadata
        document.getElementById('ocrModel').textContent = data.model || 'N/A';
        document.getElementById('ocrFileType').textContent = (data.file_type || 'N/A').toUpperCase();
        
        // Handle PDF results (multiple pages)
        if (data.file_type === 'pdf' && data.pages) {
            document.getElementById('ocrPages').style.display = 'block';
            document.getElementById('ocrTotalPages').textContent = data.total_pages || data.pages.length;
            
            // Combine all pages with page numbers
            let combinedText = '';
            data.pages.forEach((page, index) => {
                combinedText += `## Page ${page.page}\n\n${page.text}\n\n---\n\n`;
            });
            
            // Render markdown/HTML
            const html = renderMarkdown(combinedText);
            document.getElementById('ocrText').innerHTML = html;
        } else {
            // Single image result
            document.getElementById('ocrPages').style.display = 'none';
            
            // Render markdown/HTML - handles both markdown and HTML content
            const text = data.text || '';
            const html = renderMarkdown(text);
            document.getElementById('ocrText').innerHTML = html;
        }
        
        showNotification('OCR processing completed successfully', 'success');
        
    } catch (error) {
        loadingDiv.style.display = 'none';
        errorDiv.style.display = 'flex';
        errorDiv.querySelector('.error-message').textContent = error.message;
        showNotification('OCR processing failed', 'error');
        console.error('OCR Error:', error);
    }
}

// Translation Processing
async function processTranslation() {
    const sourceText = document.getElementById('sourceText').value.trim();
    const srcLang = document.getElementById('srcLang').value;
    const targetLang = document.getElementById('targetLang').value;
    
    // Validation
    if (!sourceText) {
        showNotification('Please enter text to translate', 'error');
        return;
    }
    
    if (!srcLang) {
        showNotification('Please select a source language', 'error');
        document.getElementById('srcLang').focus();
        return;
    }
    
    if (!targetLang) {
        showNotification('Please select a target language', 'error');
        document.getElementById('targetLang').focus();
        return;
    }
    
    if (srcLang === targetLang) {
        showNotification('Source and target languages must be different', 'error');
        return;
    }
    
    const loadingDiv = document.getElementById('translationLoading');
    const resultsDiv = document.getElementById('translationResults');
    const errorDiv = document.getElementById('translationError');
    
    // Show loading, hide results and errors
    loadingDiv.style.display = 'block';
    resultsDiv.style.display = 'none';
    errorDiv.style.display = 'none';
    
    try {
        const requestBody = {
            text: sourceText,
            src_lang: srcLang,
            target_lang: targetLang
        };
        
        const response = await fetch(`${apiUrl}/translation/translate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Hide loading, show results
        loadingDiv.style.display = 'none';
        resultsDiv.style.display = 'block';
        
        // Scroll to results
        resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        
        // Display metadata
        document.getElementById('translationModel').textContent = data.model || 'N/A';
        document.getElementById('translationSrcLang').textContent = (data.src_lang || 'N/A').toUpperCase();
        document.getElementById('translationTargetLang').textContent = (data.target_lang || 'N/A').toUpperCase();
        
        // Render translation as markdown/HTML
        const translation = data.translation || '';
        const html = renderMarkdown(translation);
        document.getElementById('translationText').innerHTML = html;
        
        showNotification('Translation completed successfully', 'success');
        
    } catch (error) {
        loadingDiv.style.display = 'none';
        errorDiv.style.display = 'flex';
        errorDiv.querySelector('.error-message').textContent = error.message;
        showNotification('Translation failed', 'error');
        console.error('Translation Error:', error);
    }
}

