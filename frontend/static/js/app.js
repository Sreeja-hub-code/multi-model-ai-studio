/* ── Shared UI utilities for all pages ──────────────────────────────── */

function setLoading(outputId, btnId, message = 'Processing…') {
  const out = document.getElementById(outputId);
  if (out) {
    out.className = 'output-box';
    out.innerHTML = `<div class="loader"><div class="spinner"></div>${message}</div>`;
  }
  const btn = document.getElementById(btnId);
  if (btn) btn.disabled = true;
}

function showResult(outputId, text) {
  const out = document.getElementById(outputId);
  if (!out) return;
  out.className = 'output-box';
  out.textContent = text;
}

function showError(outputId, message) {
  const out = document.getElementById(outputId);
  if (!out) return;
  out.className = 'output-box error';
  out.textContent = '⚠️ Error: ' + message;
}

function showEmpty(outputId) {
  const out = document.getElementById(outputId);
  if (!out) return;
  out.className = 'output-box empty';
  out.textContent = out.dataset.placeholder || 'Output will appear here…';
}

function resetBtn(btnId, label) {
  const btn = document.getElementById(btnId);
  if (btn) {
    btn.disabled = false;
    btn.textContent = label;
  }
}

function clearAll(inputId, outputId) {
  const inp = document.getElementById(inputId);
  if (inp) { if (inp.tagName === 'TEXTAREA' || inp.tagName === 'INPUT') inp.value = ''; }
  showEmpty(outputId);
}

function copyOutput(outputId) {
  const out = document.getElementById(outputId);
  if (!out || !out.textContent.trim()) return;
  navigator.clipboard.writeText(out.textContent).then(() => {
    const orig = out.dataset.copyLabel;
    const btn = document.querySelector(`[onclick="copyOutput('${outputId}')"]`);
    if (btn) { btn.textContent = '✅ Copied!'; setTimeout(() => btn.textContent = '📋 Copy', 1500); }
  });
}

// File preview helper
function previewFile(inputId, previewId, dropId) {
  const file = document.getElementById(inputId).files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  const preview = document.getElementById(previewId);
  if (preview) { preview.src = url; preview.style.display = 'block'; }
  const drop = document.getElementById(dropId);
  if (drop) drop.style.display = 'none';
  const changeBtn = document.getElementById('change-btn');
  if (changeBtn) changeBtn.style.display = 'inline-flex';
}

function resetFile(inputId, previewId, dropId, changeBtnId) {
  const input = document.getElementById(inputId);
  if (input) input.value = '';
  const preview = document.getElementById(previewId);
  if (preview) { preview.style.display = 'none'; preview.src = ''; }
  const drop = document.getElementById(dropId);
  if (drop) drop.style.display = 'flex';
  const btn = document.getElementById(changeBtnId);
  if (btn) btn.style.display = 'none';
}

// Drag-and-drop helpers
function handleDragOver(e) {
  e.preventDefault();
  e.currentTarget.classList.add('drag-over');
}

function handleDrop(e, inputId, previewId, dropId) {
  e.preventDefault();
  e.currentTarget.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (!file) return;
  const input = document.getElementById(inputId);
  const dt = new DataTransfer();
  dt.items.add(file);
  input.files = dt.files;
  previewFile(inputId, previewId, dropId);
}
