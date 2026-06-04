const video = document.getElementById('camera');
const canvas = document.getElementById('canvas');
const output = document.getElementById('output');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const attentionToggle = document.getElementById('attentionToggle');
const modelStatus = document.getElementById('modelStatus');
let stream = null;
let timer = null;
let busy = false;

function setMetrics(result) {
  document.getElementById('state').textContent = result.state ?? '—';
  document.getElementById('confidence').textContent = result.confidence ? result.confidence.toFixed(2) : '0.00';
  document.getElementById('risk').textContent = `${(result.risk_score ?? 0).toFixed(2)} / ${result.risk_level ?? '—'}`;
  document.getElementById('warning').textContent = result.warning ? 'Да' : 'Нет';
}

async function sendFrame() {
  if (!stream || busy || video.videoWidth === 0) return;
  busy = true;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const image = canvas.toDataURL('image/jpeg', 0.82);
  const form = new FormData();
  form.append('image_base64', image);
  form.append('attention', attentionToggle.checked ? 'true' : 'false');
  try {
    const response = await fetch('/api/frame', { method: 'POST', body: form });
    const data = await response.json();
    if (!data.ok) throw new Error(data.error || 'Ошибка анализа кадра');
    output.src = data.image;
    setMetrics(data.result);
    modelStatus.textContent = 'Модель: анализ выполняется';
  } catch (e) {
    modelStatus.textContent = 'Ошибка: ' + e.message;
  } finally {
    busy = false;
  }
}

startBtn.onclick = async () => {
  stream = await navigator.mediaDevices.getUserMedia({ video: { width: 960, height: 540 }, audio: false });
  video.srcObject = stream;
  timer = setInterval(sendFrame, 450);
};

stopBtn.onclick = () => {
  if (timer) clearInterval(timer);
  timer = null;
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null;
  modelStatus.textContent = 'Модель: остановлено';
};

document.getElementById('videoForm').onsubmit = async (event) => {
  event.preventDefault();
  const file = document.getElementById('videoFile').files[0];
  const report = document.getElementById('videoReport');
  if (!file) { report.textContent = 'Выберите видеофайл.'; return; }
  const form = new FormData();
  form.append('file', file);
  form.append('attention', attentionToggle.checked ? 'true' : 'false');
  report.textContent = 'Видео анализируется...';
  try {
    const response = await fetch('/api/video', { method: 'POST', body: form });
    const data = await response.json();
    if (!data.ok) throw new Error(data.error || 'Ошибка анализа видео');
    const videoLink = data.video_url
      ? `\n\nГотовое видео с зонами внимания: ${window.location.origin}${data.video_url}`
      : '';
    report.textContent = JSON.stringify(data.report, null, 2) + videoLink;
  } catch (e) {
    report.textContent = 'Ошибка: ' + e.message;
  }
};
