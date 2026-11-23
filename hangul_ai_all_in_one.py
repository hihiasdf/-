# -*- coding: utf-8 -*-
"""
한글 손글씨 AI - 단일 파일 실행판

기능:
- Flask 웹 그림판, 학습, 취소, 진행률, 검증 결과, 혼동 행렬, 인식
- 48x48 입력 및 학습/추론 공통 전처리
- 공간 정보 보존 CNN
- CUDA 자동 감지: RTX 2060 등 NVIDIA GPU에서 FP16 AMP 자동 사용
- CUDA가 없거나 CPU 전용 PyTorch이면 CPU로 자동 대체

필수 패키지 설치:
  python -m pip install flask numpy pillow
  PyTorch는 https://pytorch.org/get-started/locally/ 에서
  Windows / Pip / Python / CUDA를 선택해 설치 명령을 복사하세요.

실행:
  python hangul_ai_all_in_one.py
브라우저:
  http://127.0.0.1:5000

데이터 파일명:
  handwriting_dataset/data/가_1.png, 다_1.png, 라_1.png ...

GPU가 잡히지 않을 때:
  python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
CPU 강제 사용:
  set HANGUL_FORCE_CPU=1  (Windows CMD)
"""

HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>한글 손글씨 AI</title>
  <style>
    :root { color-scheme: light; --ink:#172033; --muted:#64748b; --line:#dbe3ee; --blue:#2563eb; --page:#f3f6fb; }
    * { box-sizing:border-box; }
    body { margin:0; padding:24px 14px; background:var(--page); color:var(--ink); font-family:system-ui,-apple-system,"Segoe UI",sans-serif; }
    main { width:min(100%,720px); margin:0 auto; padding:26px; background:#fff; border:1px solid var(--line); border-radius:18px; box-shadow:0 10px 30px rgba(15,23,42,.06); }
    h1 { margin:0 0 6px; text-align:center; font-size:25px; }
    .sub { margin:0 0 22px; color:var(--muted); text-align:center; font-size:13px; line-height:1.5; }
    section { margin-top:22px; }
    h2 { margin:0 0 10px; font-size:16px; }
    .canvas-wrap { display:flex; justify-content:center; margin:14px 0; }
    canvas { width:280px; height:280px; max-width:100%; background:#000; border:3px solid #334155; border-radius:12px; cursor:crosshair; touch-action:none; display:block; }
    .hint,.help { color:var(--muted); font-size:12px; line-height:1.55; }
    .hint { text-align:center; margin:7px 0 12px; }
    .row { display:flex; gap:10px; flex-wrap:wrap; }
    button { min-height:42px; padding:10px 15px; border:0; border-radius:9px; background:var(--blue); color:#fff; font-weight:700; cursor:pointer; }
    button.secondary { background:#64748b; } button.warning { background:#d97706; } button.danger { background:#dc2626; }
    button:disabled { opacity:.55; cursor:not-allowed; }
    .row button { flex:1; }
    .field { margin:12px 0; }
    label { display:block; margin-bottom:6px; font-size:13px; font-weight:700; }
    input,select { width:100%; min-height:41px; padding:9px 11px; border:1px solid #cbd5e1; border-radius:8px; background:#fff; font-size:14px; }
    .check { display:flex; align-items:flex-start; gap:9px; margin:12px 0; font-size:13px; line-height:1.5; }
    .check input { width:auto; min-height:0; margin-top:3px; }
    .panel { margin-top:13px; padding:14px; border:1px solid var(--line); border-radius:10px; background:#f8fafc; font-size:13px; line-height:1.55; overflow-wrap:anywhere; }
    .hidden { display:none !important; }
    .progress { height:9px; margin-top:10px; background:#e2e8f0; border-radius:30px; overflow:hidden; }
    .bar { width:0; height:100%; background:var(--blue); transition:width .2s; }
    .result { margin-top:12px; text-align:center; font-size:22px; font-weight:800; }
    .note { margin-top:6px; color:var(--muted); text-align:center; font-size:12px; }
    .top { display:grid; gap:7px; margin-top:12px; }
    .top div { display:flex; justify-content:space-between; padding:9px 10px; border:1px solid var(--line); border-radius:8px; background:#fff; }
    .metrics { margin-top:14px; }
    .metric-cards { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; margin:9px 0; }
    .metric-card { padding:10px; background:#fff; border:1px solid var(--line); border-radius:8px; }
    .metric-card strong { display:block; font-size:19px; }
    .metric-card span { color:var(--muted); font-size:11px; }
    table { width:100%; border-collapse:collapse; margin-top:8px; font-size:12px; }
    th,td { padding:7px 5px; text-align:center; border-bottom:1px solid var(--line); }
    th { background:#eef2f7; }
    .table-wrap { overflow-x:auto; }
    .muted { color:var(--muted); }
    @media(max-width:540px) { main { padding:18px 14px; } h1 { font-size:21px; } .row button { flex-basis:100%; } canvas { width:250px; height:250px; } }
  </style>
</head>
<body>
<main>
  <h1>✍️ 한글 손글씨 AI</h1>
  <p class="sub">48×48 학습 데이터에 맞춘 공간 정보 보존 CNN<br>CUDA GPU 자동 감지 · CUDA에서는 FP16 혼합 정밀도 적용</p>
  <div class="panel" id="deviceInfo">실행 장치 확인 중...</div>

  <section>
    <h2>1. 인식할 글자 그리기</h2>
    <div class="canvas-wrap"><canvas id="drawCanvas" width="256" height="256" aria-label="글자 그리기"></canvas></div>
    <p class="hint">검은 바탕에 흰색으로 글자 하나를 크게 써 주세요.</p>
    <div class="row">
      <button class="secondary" id="clearBtn" type="button">지우기</button>
      <button id="predictBtn" type="button">🔍 글자 인식</button>
    </div>
    <div id="predictionBox" class="panel hidden" aria-live="polite">
      <div id="predictionMain" class="result"></div>
      <div id="predictionNote" class="note"></div>
      <div id="topList" class="top"></div>
    </div>
  </section>

  <section>
    <h2>2. 학습 설정</h2>
    <div class="field">
      <label for="subfolderInput">데이터 하위 폴더</label>
      <input id="subfolderInput" type="text" value="data" placeholder="예: data">
      <p class="help">기본 위치: handwriting_dataset/data</p>
    </div>
    <div class="field">
      <label for="maxIndexInput">글자당 최대 파일 번호 (비우면 전체 사용)</label>
      <input id="maxIndexInput" type="number" min="1" placeholder="전체 이미지 사용">
    </div>
    <div class="field">
      <label for="epochsInput">최대 학습 Epoch</label>
      <input id="epochsInput" type="number" min="1" max="200" value="60">
      <p class="help">검증 성능이 일정 기간 개선되지 않으면 조기 종료합니다.</p>
    </div>
    <div class="field">
      <label for="trainModeSelect">학습 모드</label>
      <select id="trainModeSelect">
        <option value="custom" selected>특정 글자만 학습</option>
        <option value="all">모든 글자 학습</option>
      </select>
    </div>
    <div class="field" id="customGroup">
      <label for="customCharsInput">학습할 글자 (쉼표 또는 공백으로 구분)</label>
      <input id="customCharsInput" type="text" value="가, 나, 다, 라, 앍">
    </div>
    <label class="check">
      <input id="groupByIndexInput" type="checkbox" checked>
      <span><strong>파일 번호가 같으면 같은 글꼴/작성자 그룹으로 분리</strong><br><span class="muted">예: 다_1.png와 나_1.png를 같은 검증 그룹에 둡니다. 번호가 같은 폰트·작성자 샘플이라면 검증 데이터 누수를 줄이는 데 유리합니다.</span></span>
    </label>
    <div class="row">
      <button class="warning" id="trainBtn" type="button">🚀 모델 학습 시작</button>
      <button class="danger hidden" id="cancelBtn" type="button">학습 취소</button>
    </div>
    <div class="panel" aria-live="polite">
      <div id="statusMessage">상태: 대기 중</div>
      <div class="progress"><div id="progressBar" class="bar"></div></div>
      <div id="metricsBox" class="metrics hidden"></div>
    </div>
  </section>
  <p class="help" style="margin-top:18px">주의: 화면의 점수는 모델의 softmax 출력이며 실제 정확도와 같지 않습니다. 실제 손글씨 인식률은 학습 데이터가 실제 손글씨와 얼마나 비슷한지에 크게 좌우됩니다.</p>
</main>
<script>
  const canvas = document.getElementById('drawCanvas');
  const ctx = canvas.getContext('2d');
  const statusMessage = document.getElementById('statusMessage');
  const progressBar = document.getElementById('progressBar');
  let drawing = false;
  let statusTimer = null;

  function clearCanvas() {
    ctx.save(); ctx.fillStyle = '#000'; ctx.fillRect(0, 0, canvas.width, canvas.height); ctx.restore();
    document.getElementById('predictionBox').classList.add('hidden');
  }
  clearCanvas();
  // 256px에서 부드럽게 그린 후 서버가 동일한 방식으로 48px로 정규화합니다.
  ctx.lineWidth = 15; ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.strokeStyle = '#fff';

  function pointFromEvent(event) {
    const rect = canvas.getBoundingClientRect();
    return { x:(event.clientX-rect.left)*canvas.width/rect.width, y:(event.clientY-rect.top)*canvas.height/rect.height };
  }
  canvas.addEventListener('pointerdown', (event) => {
    event.preventDefault(); drawing = true; canvas.setPointerCapture(event.pointerId);
    const p = pointFromEvent(event); ctx.beginPath(); ctx.arc(p.x,p.y,ctx.lineWidth/2,0,Math.PI*2); ctx.fillStyle='#fff'; ctx.fill(); ctx.beginPath(); ctx.moveTo(p.x,p.y);
  });
  canvas.addEventListener('pointermove', (event) => {
    if (!drawing) return; event.preventDefault(); const p=pointFromEvent(event); ctx.lineTo(p.x,p.y); ctx.stroke(); ctx.beginPath(); ctx.moveTo(p.x,p.y);
  });
  function stopDrawing(){ drawing=false; ctx.beginPath(); }
  canvas.addEventListener('pointerup',stopDrawing); canvas.addEventListener('pointercancel',stopDrawing); canvas.addEventListener('lostpointercapture',stopDrawing);
  document.getElementById('clearBtn').addEventListener('click',clearCanvas);
  document.getElementById('trainModeSelect').addEventListener('change',(e)=>{
    document.getElementById('customGroup').classList.toggle('hidden',e.target.value==='all');
  });

  async function jsonRequest(url, options={}) {
    const response=await fetch(url,options); const data=await response.json().catch(()=>({}));
    if(!response.ok) throw new Error(data.message || `요청 실패 (${response.status})`); return data;
  }
  function setTrainingUI(active){
    document.getElementById('trainBtn').classList.toggle('hidden',active);
    document.getElementById('cancelBtn').classList.toggle('hidden',!active);
    document.getElementById('trainBtn').disabled=active;
  }
  function renderMetrics(metrics){
    const box=document.getElementById('metricsBox'); box.replaceChildren();
    if(!metrics){box.classList.add('hidden');return;}
    box.classList.remove('hidden');
    const cards=document.createElement('div'); cards.className='metric-cards';
    for(const [label,value] of [['검증 정확도',`${metrics.validation_accuracy}%`],['클래스 평균 재현율',`${metrics.macro_recall}%`]]){
      const card=document.createElement('div'); card.className='metric-card'; const strong=document.createElement('strong'); strong.textContent=value; const span=document.createElement('span'); span.textContent=label; card.append(strong,span); cards.append(card);
    }
    box.append(cards);
    const title=document.createElement('strong'); title.textContent='글자별 검증 결과'; box.append(title);
    const wrap=document.createElement('div'); wrap.className='table-wrap'; const table=document.createElement('table');
    const thead=document.createElement('thead'); const hr=document.createElement('tr');
    for(const t of ['글자','정밀도','재현율','검증 장수']){const th=document.createElement('th');th.textContent=t;hr.append(th);} thead.append(hr);table.append(thead);
    const tbody=document.createElement('tbody');
    for(const item of (metrics.per_class||[])){const tr=document.createElement('tr');for(const v of [item.character,`${item.precision}%`,`${item.recall}%`,item.support]){const td=document.createElement('td');td.textContent=v;tr.append(td);}tbody.append(tr);}
    table.append(tbody);wrap.append(table);box.append(wrap);
    if((metrics.classes||[]).length<=12){
      const subtitle=document.createElement('p');subtitle.className='help';subtitle.textContent='혼동 행렬: 행은 실제 글자, 열은 모델 예측 글자';box.append(subtitle);
      const matrixWrap=document.createElement('div');matrixWrap.className='table-wrap';const matrix=document.createElement('table');
      const mhead=document.createElement('thead');const mhr=document.createElement('tr');
      for(const v of ['실제 / 예측',...(metrics.classes||[])]){const th=document.createElement('th');th.textContent=v;mhr.append(th);}mhead.append(mhr);matrix.append(mhead);
      const mb=document.createElement('tbody');(metrics.confusion_matrix||[]).forEach((row,i)=>{const tr=document.createElement('tr');const label=document.createElement('th');label.textContent=metrics.classes[i];tr.append(label);row.forEach(n=>{const td=document.createElement('td');td.textContent=n;tr.append(td);});mb.append(tr);});matrix.append(mb);matrixWrap.append(matrix);box.append(matrixWrap);
    }
  }
  async function pollStatus(){
    try{const data=await jsonRequest('/status',{cache:'no-store'});statusMessage.textContent=data.progress_message||'상태 확인 중';progressBar.style.width=`${Math.max(0,Math.min(100,Number(data.progress)||0))}%`;setTrainingUI(Boolean(data.is_training));renderMetrics(data.metrics);
      if(data.device_info){const d=data.device_info;const el=document.getElementById('deviceInfo');el.textContent=d.device==='cuda'?`학습 장치: ${d.device_name} | VRAM: ${d.gpu_memory_gb??'확인 불가'}GB | CUDA 활성화 | FP16 AMP 사용`:`학습 장치: ${d.device_name} | CUDA 사용 불가 또는 비활성화 | CPU/MPS 정밀도 사용`;}
      if(!data.is_training && statusTimer!==null){clearInterval(statusTimer);statusTimer=null;}
    }catch(error){statusMessage.textContent=`상태 확인 오류: ${error.message}`;}
  }

  document.getElementById('trainBtn').addEventListener('click',async()=>{
    const mode=document.getElementById('trainModeSelect').value;
    const rawMax=document.getElementById('maxIndexInput').value.trim();
    const epochs=Number(document.getElementById('epochsInput').value||60);
    const customChars=mode==='custom'?document.getElementById('customCharsInput').value.split(/[,\s]+/u).filter(Boolean):null;
    if(mode==='custom'&&(!customChars||!customChars.length)){statusMessage.textContent='학습할 글자를 입력하세요.';return;}
    if(rawMax&&(!Number.isInteger(Number(rawMax))||Number(rawMax)<1)){statusMessage.textContent='최대 번호는 1 이상의 정수 또는 빈칸이어야 합니다.';return;}
    if(!Number.isInteger(epochs)||epochs<1||epochs>200){statusMessage.textContent='Epoch는 1~200 사이로 입력하세요.';return;}
    setTrainingUI(true);progressBar.style.width='0%';renderMetrics(null);statusMessage.textContent='학습 준비 중...';
    try{
      await jsonRequest('/train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
        subfolder:document.getElementById('subfolderInput').value.trim()||'data',
        maxIndex:rawMax?Number(rawMax):null,epochs,customChars,
        groupByIndex:document.getElementById('groupByIndexInput').checked
      })});
      if(statusTimer!==null)clearInterval(statusTimer);statusTimer=setInterval(pollStatus,1000);pollStatus();
    }catch(error){statusMessage.textContent=`학습 시작 오류: ${error.message}`;setTrainingUI(false);}
  });
  document.getElementById('cancelBtn').addEventListener('click',async()=>{
    try{const data=await jsonRequest('/cancel',{method:'POST'});statusMessage.textContent=data.message||'취소 요청을 보냈습니다.';}
    catch(error){statusMessage.textContent=`취소 오류: ${error.message}`;} pollStatus();
  });
  document.getElementById('predictBtn').addEventListener('click',async()=>{
    const button=document.getElementById('predictBtn');const box=document.getElementById('predictionBox');
    const main=document.getElementById('predictionMain');const note=document.getElementById('predictionNote');const top=document.getElementById('topList');
    button.disabled=true;box.classList.remove('hidden');main.textContent='인식 중...';note.textContent='';top.replaceChildren();
    try{
      const data=await jsonRequest('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:canvas.toDataURL('image/png')})});
      main.textContent=`인식 결과: ${data.prediction} (${data.confidence}점)`;note.textContent=data.note||'';
      (data.top_predictions||[]).forEach((item,i)=>{const row=document.createElement('div');const name=document.createElement('span');const score=document.createElement('strong');name.textContent=`${i+1}위 · ${item.character}`;score.textContent=`${item.score}점`;row.append(name,score);top.append(row);});
    }catch(error){main.textContent=`인식 실패: ${error.message}`;}
    finally{button.disabled=false;}
  });
  pollStatus();
  jsonRequest('/status',{cache:'no-store'}).then(data=>{if(data.is_training&&statusTimer===null)statusTimer=setInterval(pollStatus,1000);}).catch(()=>{});
</script>
</body>
</html>

"""


import base64
import copy
import io
import os
import random
import re
import threading
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, jsonify, render_template_string, request
from PIL import Image, ImageOps, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# ============================================================
# 기본 설정
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "handwriting_dataset"
MODEL_PATH = BASE_DIR / "handwriting_model.pth"
TEMP_MODEL_PATH = BASE_DIR / "handwriting_model.tmp.pth"

# 원본 데이터가 48x48이므로 확대하지 않고 원본 해상도 사용
IMAGE_SIZE = 48
ARCH_VERSION = 3
ARCH_NAME = "HangulCNN-Spatial-48-v3"
DEFAULT_EPOCHS = 60
MAX_EPOCHS = 200
BATCH_SIZE = 32
VALIDATION_RATIO = 0.20
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 12
MIN_IMAGES_PER_CLASS = 3
MAX_UPLOAD_BYTES = 5 * 1024 * 1024

# HANGUL_FORCE_CPU=1 환경 변수를 주면 GPU가 있어도 CPU로 실행할 수 있습니다.
FORCE_CPU = os.getenv("HANGUL_FORCE_CPU", "0") == "1"
if not FORCE_CPU and torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
elif not FORCE_CPU and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# RTX 2060은 FP16 Tensor Core를 지원하므로 CUDA에서 자동 혼합 정밀도를 사용합니다.
# CPU/MPS에서는 AMP를 끄고 기존 FP32 학습을 사용합니다.
USE_AMP = DEVICE.type == "cuda"
PIN_MEMORY = DEVICE.type == "cuda"

if DEVICE.type == "cuda":
    GPU_NAME = torch.cuda.get_device_name(DEVICE)
    try:
        GPU_MEMORY_GB = round(torch.cuda.get_device_properties(DEVICE).total_memory / (1024 ** 3), 1)
    except Exception:
        GPU_MEMORY_GB = None
else:
    GPU_NAME = ""
    GPU_MEMORY_GB = None

# CPU에서 지나친 스레드 경쟁을 피한다. GPU에는 영향을 주지 않는다.
if DEVICE.type == "cpu":
    try:
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    except RuntimeError:
        pass


def autocast_context():
    """CUDA에서는 FP16 AMP, CPU/MPS에서는 일반 정밀도를 사용합니다."""
    if USE_AMP:
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
    from contextlib import nullcontext
    return nullcontext()


def create_grad_scaler():
    """PyTorch 2.2+ API 차이를 고려해 GradScaler를 생성합니다."""
    try:
        return torch.amp.GradScaler("cuda", enabled=USE_AMP)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=USE_AMP)


def get_device_info() -> dict:
    return {
        "device": str(DEVICE),
        "device_name": GPU_NAME if DEVICE.type == "cuda" else ("Apple Metal (MPS)" if DEVICE.type == "mps" else "CPU"),
        "cuda_available": bool(torch.cuda.is_available()),
        "amp_enabled": bool(USE_AMP),
        "gpu_memory_gb": GPU_MEMORY_GB,
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "forced_cpu": FORCE_CPU,
    }


print("=" * 60)
print(f"PyTorch 버전: {torch.__version__}")
print(f"실행 장치: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {GPU_NAME} | VRAM: {GPU_MEMORY_GB or '확인 불가'} GB | AMP(FP16): 사용")
else:
    print("CUDA GPU를 사용할 수 없어 CPU/MPS로 실행합니다.")
    if not FORCE_CPU and torch.version.cuda is None:
        print("안내: 현재 설치된 PyTorch가 CPU 전용 빌드일 수 있습니다.")
print("=" * 60)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES + 1024 * 1024
DATA_DIR.mkdir(parents=True, exist_ok=True)

FILENAME_PATTERN = re.compile(r"^(.+?)_(\d+)\.(png|jpe?g)$", re.IGNORECASE)
RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS

status_lock = threading.Lock()
training_start_lock = threading.Lock()
model_cache_lock = threading.Lock()
training_status = {
    "is_training": False,
    "cancel_requested": False,
    "progress": 0,
    "progress_message": "대기 중",
    "last_error": None,
    "metrics": None,
    "class_counts": None,
    "split_strategy": None,
}
_model_cache = {"mtime": None, "model": None, "classes": None, "metrics": None}


def seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_training_status(**updates) -> None:
    with status_lock:
        training_status.update(updates)


def get_training_status() -> dict:
    with status_lock:
        return copy.deepcopy(training_status)


def is_cancel_requested() -> bool:
    with status_lock:
        return bool(training_status["cancel_requested"])


# ============================================================
# 전처리: 학습과 예측에서 반드시 동일하게 사용
# 검은 배경/흰 글씨, 흰 배경/검은 글씨 모두 지원
# ============================================================
def preprocess_handwriting(image: Image.Image, size: int = IMAGE_SIZE, margin: int = 5) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("L")
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("비어 있는 이미지입니다.")

    # 테두리의 중앙값으로 배경색을 추정한다.
    border = np.concatenate((arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]))
    bg = float(np.median(border))
    if bg >= 127:
        foreground = bg - arr.astype(np.float32)
    else:
        foreground = arr.astype(np.float32) - bg
    foreground = np.clip(foreground, 0, 255).astype(np.uint8)

    # 아주 약한 압축 노이즈는 글자 영역으로 취급하지 않는다.
    mask = foreground >= 18
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return Image.new("L", (size, size), 0)

    # 경계에 한 픽셀 여유를 둔다.
    left = max(0, int(xs.min()) - 1)
    right = min(image.width, int(xs.max()) + 2)
    top = max(0, int(ys.min()) - 1)
    bottom = min(image.height, int(ys.max()) + 2)
    crop = Image.fromarray(foreground).crop((left, top, right, bottom))

    # 종횡비를 유지하고 중앙에 배치한다. 가로/세로를 각각 늘리지 않는다.
    target = max(1, size - margin * 2)
    w, h = crop.size
    scale = min(target / max(w, 1), target / max(h, 1))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    crop = crop.resize((nw, nh), RESAMPLE)

    canvas = Image.new("L", (size, size), 0)
    canvas.paste(crop, ((size - nw) // 2, (size - nh) // 2))
    return canvas


# ============================================================
# 데이터셋
# 파일명 예시: 다_1.png, 나_25.jpg
# 숫자 suffix는 같은 글꼴/작성자/샘플 묶음일 수 있으므로 그룹 분할 지원
# ============================================================
class HandwritingDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        subfolder: str = "data",
        max_index: Optional[int] = None,
        custom_chars: Optional[list[str]] = None,
        transform=None,
    ):
        self.transform = transform
        self.image_paths: list[Path] = []
        self.raw_labels: list[str] = []
        self.sample_ids: list[int] = []

        data_root = data_dir.resolve()
        requested_dir = (data_root / subfolder).resolve() if subfolder else data_root
        try:
            requested_dir.relative_to(data_root)
        except ValueError as exc:
            raise ValueError("하위 폴더는 handwriting_dataset 내부 경로만 사용할 수 있습니다.") from exc

        target_dir = requested_dir if requested_dir.is_dir() else data_root
        if not target_dir.is_dir():
            raise FileNotFoundError(f"데이터 폴더를 찾을 수 없습니다: {target_dir}")

        wanted = None
        if custom_chars:
            wanted = {unicodedata.normalize("NFC", str(x).strip()) for x in custom_chars if str(x).strip()}

        items = []
        for current_root, dirs, files in os.walk(target_dir):
            dirs.sort()
            for filename in sorted(files):
                match = FILENAME_PATTERN.match(filename)
                if not match:
                    continue
                label = unicodedata.normalize("NFC", match.group(1).strip())
                sample_id = int(match.group(2))
                if label.lower() == "sample":
                    continue
                if wanted is not None and label not in wanted:
                    continue
                if max_index is not None and sample_id > max_index:
                    continue
                items.append((Path(current_root) / filename, label, sample_id))

        items.sort(key=lambda x: (x[1], x[2], str(x[0])))
        self.image_paths = [x[0] for x in items]
        self.raw_labels = [x[1] for x in items]
        self.sample_ids = [x[2] for x in items]
        self.classes = sorted(set(self.raw_labels))
        self.class_to_idx = {label: i for i, label in enumerate(self.classes)}
        self.labels = [self.class_to_idx[label] for label in self.raw_labels]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        try:
            with Image.open(path) as im:
                image = preprocess_handwriting(im, size=IMAGE_SIZE)
        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"이미지를 읽을 수 없습니다: {path.name}") from exc
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]


def split_by_sample_id(dataset: HandwritingDataset, validation_ratio: float = VALIDATION_RATIO):
    """동일한 suffix가 여러 글자에서 같은 스타일 ID라면 함께 검증셋으로 보낸다."""
    groups = defaultdict(list)
    for i, sample_id in enumerate(dataset.sample_ids):
        groups[sample_id].append(i)

    group_ids = list(groups)
    if len(group_ids) < 2:
        return None

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(group_ids)
    val_group_count = max(1, int(round(len(group_ids) * validation_ratio)))
    val_group_count = min(val_group_count, len(group_ids) - 1)
    val_groups = set(group_ids[:val_group_count])

    train_indices, val_indices = [], []
    for group_id, indices in groups.items():
        (val_indices if group_id in val_groups else train_indices).extend(indices)

    # 모든 클래스가 학습과 검증에 모두 등장하는지 검사한다.
    train_classes = {dataset.labels[i] for i in train_indices}
    val_classes = {dataset.labels[i] for i in val_indices}
    expected = set(range(len(dataset.classes)))
    if train_classes == expected and val_classes == expected:
        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        return train_indices, val_indices, "파일 번호 그룹 분리(동일 번호는 같은 그룹)"
    return None


def stratified_split(dataset: HandwritingDataset, validation_ratio: float = VALIDATION_RATIO):
    groups = defaultdict(list)
    for i, label in enumerate(dataset.labels):
        groups[label].append(i)

    rng = random.Random(RANDOM_SEED)
    train_indices, val_indices = [], []
    for label, indices in groups.items():
        indices = list(indices)
        rng.shuffle(indices)
        val_count = max(1, int(round(len(indices) * validation_ratio)))
        val_count = min(val_count, len(indices) - 1)
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices, "클래스별 층화 분리"


# ============================================================
# 작지만 공간 정보를 유지하는 CNN
# 입력 48x48 -> 24x24 -> 12x12 -> 6x6
# 6x6 특징 맵을 1x1로 평균내지 않고 위치 정보를 보존
# ============================================================
def conv_block(in_channels: int, out_channels: int, two_convs: bool = True) -> nn.Sequential:
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.GELU(),
    ]
    if two_convs:
        layers.extend([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        ])
    layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class HandwritingCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32, two_convs=True),       # 48 -> 24
            conv_block(32, 64, two_convs=True),     # 24 -> 12
            conv_block(64, 128, two_convs=True),    # 12 -> 6
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def evaluate_model(model, loader, criterion, classes):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    class_count = len(classes)
    confusion = np.zeros((class_count, class_count), dtype=np.int64)

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(DEVICE, non_blocking=(DEVICE.type == "cuda"))
            targets = targets.to(DEVICE, non_blocking=(DEVICE.type == "cuda"))
            with autocast_context():
                logits = model(images)
                loss = criterion(logits, targets)
            predictions = logits.argmax(dim=1)
            batch = int(targets.size(0))
            total_loss += float(loss.item()) * batch
            total += batch
            correct += int((predictions == targets).sum().item())
            for actual, predicted in zip(targets.cpu().tolist(), predictions.cpu().tolist()):
                confusion[actual, predicted] += 1

    per_class = []
    recalls = []
    for i, label in enumerate(classes):
        support = int(confusion[i].sum())
        true_positive = int(confusion[i, i])
        predicted_count = int(confusion[:, i].sum())
        recall = true_positive / support if support else 0.0
        precision = true_positive / predicted_count if predicted_count else 0.0
        if support:
            recalls.append(recall)
        per_class.append({
            "character": label,
            "precision": round(precision * 100, 1),
            "recall": round(recall * 100, 1),
            "support": support,
        })

    accuracy = correct / max(total, 1)
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    metrics = {
        "validation_accuracy": round(accuracy * 100, 2),
        "macro_recall": round(macro_recall * 100, 2),
        "validation_loss": round(total_loss / max(total, 1), 5),
        "classes": classes,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }
    return total_loss / max(total, 1), macro_recall, metrics


def make_balanced_sampler(labels: list[int], num_classes: int):
    counts = np.bincount(labels, minlength=num_classes)
    positive = counts[counts > 0]
    if len(positive) == 0 or positive.max() / positive.min() < 1.5:
        return None, counts.tolist()
    # 역빈도 전체를 그대로 사용하면 소수 클래스가 과도하게 반복될 수 있어 역제곱근 가중치 사용
    class_weights = 1.0 / np.sqrt(np.maximum(counts, 1))
    sample_weights = [float(class_weights[label]) for label in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler, counts.tolist()


# ============================================================
# 학습 워커: 백그라운드 실행, 진행률/취소/조기 종료 지원
# ============================================================
def train_worker(config: dict) -> None:
    started = time.time()
    temp_path = TEMP_MODEL_PATH
    try:
        seed_everything()
        dataset = HandwritingDataset(
            DATA_DIR,
            subfolder=config["subfolder"],
            max_index=config["max_index"],
            custom_chars=config["custom_chars"],
        )

        if len(dataset) == 0:
            raise ValueError("데이터가 없습니다. 폴더와 파일 이름(예: 다_1.png)을 확인하세요.")
        if len(dataset.classes) < 2:
            raise ValueError("분류하려면 서로 다른 글자가 2개 이상 필요합니다.")

        counts_by_class = {label: dataset.labels.count(i) for i, label in enumerate(dataset.classes)}
        too_small = {label: count for label, count in counts_by_class.items() if count < MIN_IMAGES_PER_CLASS}
        if too_small:
            detail = ", ".join(f"{label}: {count}장" for label, count in too_small.items())
            raise ValueError(f"글자별 이미지가 최소 {MIN_IMAGES_PER_CLASS}장 필요합니다. 현재: {detail}")

        if config["group_by_index"]:
            split = split_by_sample_id(dataset)
        else:
            split = None
        if split is None:
            train_indices, val_indices, split_name = stratified_split(dataset)
        else:
            train_indices, val_indices, split_name = split

        if not train_indices or not val_indices:
            raise ValueError("학습/검증 데이터 분리에 실패했습니다. 글자별 이미지 수를 늘려 주세요.")

        train_transform = transforms.Compose([
            transforms.RandomAffine(
                degrees=2.0,
                translate=(0.025, 0.025),
                scale=(0.96, 1.04),
                shear=None,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            ),
            transforms.ToTensor(),
        ])
        eval_transform = transforms.ToTensor()
        train_dataset = copy.copy(dataset)
        train_dataset.transform = train_transform
        val_dataset = copy.copy(dataset)
        val_dataset.transform = eval_transform

        train_labels = [dataset.labels[i] for i in train_indices]
        sampler, train_counts = make_balanced_sampler(train_labels, len(dataset.classes))
        train_loader = DataLoader(
            Subset(train_dataset, train_indices),
            batch_size=min(BATCH_SIZE, max(1, len(train_indices))),
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=0,
            pin_memory=PIN_MEMORY,
        )
        val_loader = DataLoader(
            Subset(val_dataset, val_indices),
            batch_size=min(BATCH_SIZE, max(1, len(val_indices))),
            shuffle=False,
            num_workers=0,
            pin_memory=PIN_MEMORY,
        )

        num_classes = len(dataset.classes)
        model = HandwritingCNN(num_classes).to(DEVICE)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
        # FP16 AMP는 CUDA에서만 켭니다. CPU에서는 scaler가 비활성화됩니다.
        scaler = create_grad_scaler()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-5
        )

        set_training_status(
            progress=0,
            progress_message=(
                f"데이터 {len(dataset):,}장 | {num_classes}개 글자 | "
                f"학습 {len(train_indices):,}장 / 검증 {len(val_indices):,}장 | {split_name}"
            ),
            class_counts=counts_by_class,
            split_strategy=split_name,
            metrics=None,
        )

        best_state = None
        best_metrics = None
        best_score = -1.0
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        requested_epochs = int(config["epochs"])
        actual_epochs = 0

        for epoch in range(requested_epochs):
            if is_cancel_requested():
                set_training_status(
                    is_training=False,
                    progress_message="학습이 취소되었습니다. 이전 모델은 유지했습니다.",
                )
                return

            model.train()
            sum_loss = 0.0
            sum_correct = 0
            sum_count = 0

            for images, targets in train_loader:
                if is_cancel_requested():
                    set_training_status(
                        is_training=False,
                        progress_message="학습이 취소되었습니다. 이전 모델은 유지했습니다.",
                    )
                    return

                images = images.to(DEVICE, non_blocking=(DEVICE.type == "cuda"))
                targets = targets.to(DEVICE, non_blocking=(DEVICE.type == "cuda"))
                optimizer.zero_grad(set_to_none=True)
                with autocast_context():
                    logits = model(images)
                    loss = criterion(logits, targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()

                batch = int(targets.size(0))
                sum_loss += float(loss.item()) * batch
                sum_correct += int((logits.argmax(dim=1) == targets).sum().item())
                sum_count += batch

            train_loss = sum_loss / max(sum_count, 1)
            train_accuracy = sum_correct / max(sum_count, 1)
            val_loss, val_macro_recall, metrics = evaluate_model(model, val_loader, criterion, dataset.classes)
            scheduler.step(val_macro_recall)
            actual_epochs = epoch + 1

            # 클래스별 재현율 평균을 기준으로 best model을 선택하고, 동률이면 검증 손실을 사용
            improved = (val_macro_recall > best_score + 1e-6) or (
                abs(val_macro_recall - best_score) <= 1e-6 and val_loss < best_val_loss
            )
            if improved:
                best_score = val_macro_recall
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                best_metrics = copy.deepcopy(metrics)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            progress = int((epoch + 1) / requested_epochs * 100)
            elapsed = time.time() - started
            current_lr = optimizer.param_groups[0]["lr"]
            set_training_status(
                progress=progress,
                progress_message=(
                    f"{epoch + 1}/{requested_epochs} epoch | "
                    f"학습 loss {train_loss:.4f}, 정확도 {train_accuracy * 100:.1f}% | "
                    f"검증 정확도 {metrics['validation_accuracy']:.1f}%, "
                    f"평균 재현율 {metrics['macro_recall']:.1f}% | "
                    f"LR {current_lr:.6f} | {elapsed:.1f}초"
                ),
                metrics=metrics,
            )

            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                break

        if is_cancel_requested():
            set_training_status(
                is_training=False,
                progress_message="학습이 취소되었습니다. 이전 모델은 유지했습니다.",
            )
            return
        if best_state is None or best_metrics is None:
            raise RuntimeError("저장할 최적 모델을 만들지 못했습니다.")

        checkpoint = {
            "arch_version": ARCH_VERSION,
            "arch_name": ARCH_NAME,
            "model_state": best_state,
            "classes": dataset.classes,
            "image_size": IMAGE_SIZE,
            "preprocess_version": 2,
            "metrics": best_metrics,
            "class_counts": counts_by_class,
            "split_strategy": split_name,
            "epochs_trained": actual_epochs,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, MODEL_PATH)
        invalidate_model_cache()

        elapsed = time.time() - started
        msg = (
            f"학습 완료! {num_classes}개 글자, {len(dataset):,}장 사용 | "
            f"검증 정확도 {best_metrics['validation_accuracy']:.1f}% | "
            f"평균 재현율 {best_metrics['macro_recall']:.1f}% | "
            f"{actual_epochs} epoch / {elapsed:.1f}초"
        )
        set_training_status(
            is_training=False,
            progress=100,
            progress_message=msg,
            metrics=best_metrics,
            class_counts=counts_by_class,
            split_strategy=split_name,
            last_error=None,
        )

    except Exception as exc:
        print(f"학습 오류: {exc}")
        set_training_status(
            is_training=False,
            progress_message=f"학습 오류: {exc}",
            last_error=str(exc),
        )
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass


# ============================================================
# 모델 캐시 및 추론
# ============================================================
def invalidate_model_cache() -> None:
    with model_cache_lock:
        _model_cache.update({"mtime": None, "model": None, "classes": None, "metrics": None})


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def load_model_for_prediction():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("저장된 모델이 없습니다. 먼저 학습을 진행하세요.")

    mtime = MODEL_PATH.stat().st_mtime_ns
    with model_cache_lock:
        if _model_cache["model"] is not None and _model_cache["mtime"] == mtime:
            return _model_cache["model"], _model_cache["classes"], _model_cache["metrics"]

        checkpoint = safe_torch_load(MODEL_PATH)
        if checkpoint.get("arch_version") != ARCH_VERSION:
            raise RuntimeError("기존 모델은 이전 코드로 학습되었습니다. 새 코드에서 다시 학습해야 합니다.")
        classes = checkpoint.get("classes") or []
        if not classes:
            raise RuntimeError("모델 파일에 글자 목록이 없습니다.")
        model = HandwritingCNN(len(classes)).to(DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        _model_cache.update({
            "mtime": mtime,
            "model": model,
            "classes": classes,
            "metrics": checkpoint.get("metrics"),
        })
        return model, classes, checkpoint.get("metrics")


# ============================================================
# Flask routes
# ============================================================
@app.route("/")
def home():
    return render_template_string(HTML_PAGE)


@app.route("/status", methods=["GET"])
def status():
    payload = get_training_status()
    payload["device_info"] = get_device_info()
    return jsonify(payload)


@app.route("/train", methods=["POST"])
def start_training():
    data = request.get_json(silent=True) or {}
    subfolder = str(data.get("subfolder", "data")).strip()
    if subfolder in ("", "."):
        subfolder = ""

    max_raw = data.get("maxIndex")
    try:
        max_index = int(max_raw) if max_raw not in (None, "") else None
        if max_index is not None and max_index < 1:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "최대 번호는 1 이상의 정수 또는 빈칸이어야 합니다."}), 400

    try:
        epochs = int(data.get("epochs", DEFAULT_EPOCHS))
    except (TypeError, ValueError):
        epochs = DEFAULT_EPOCHS
    if not 1 <= epochs <= MAX_EPOCHS:
        return jsonify({"status": "error", "message": f"Epoch는 1~{MAX_EPOCHS} 범위로 입력하세요."}), 400

    custom_chars = data.get("customChars")
    if isinstance(custom_chars, str):
        custom_chars = re.split(r"[,\s]+", custom_chars.strip())
    if custom_chars is not None:
        if not isinstance(custom_chars, list):
            return jsonify({"status": "error", "message": "학습 글자 목록 형식이 올바르지 않습니다."}), 400
        custom_chars = [unicodedata.normalize("NFC", str(c).strip()) for c in custom_chars if str(c).strip()]
        if not custom_chars:
            custom_chars = None

    config = {
        "subfolder": subfolder,
        "max_index": max_index,
        "epochs": epochs,
        "custom_chars": custom_chars,
        "group_by_index": bool(data.get("groupByIndex", True)),
    }

    with training_start_lock:
        if get_training_status()["is_training"]:
            return jsonify({"status": "error", "message": "이미 학습이 진행 중입니다."}), 409
        set_training_status(
            is_training=True,
            cancel_requested=False,
            progress=0,
            progress_message="데이터를 확인하고 학습을 준비하고 있습니다.",
            last_error=None,
            metrics=None,
            class_counts=None,
            split_strategy=None,
        )

    thread = threading.Thread(target=train_worker, args=(config,), daemon=True)
    thread.start()
    return jsonify({"status": "started", "message": "학습을 시작했습니다."}), 202


@app.route("/cancel", methods=["POST"])
def cancel_training():
    if not get_training_status()["is_training"]:
        return jsonify({"status": "error", "message": "진행 중인 학습이 없습니다."}), 400
    set_training_status(cancel_requested=True, progress_message="취소 요청을 처리하고 있습니다.")
    return jsonify({"status": "success", "message": "학습 취소 요청을 보냈습니다."})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    image_data = data.get("image")
    if not isinstance(image_data, str) or not image_data:
        return jsonify({"status": "error", "message": "인식할 이미지가 전달되지 않았습니다."}), 400

    try:
        model, classes, metrics = load_model_for_prediction()
        encoded = image_data.split(",", 1)[1] if "," in image_data else image_data
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > MAX_UPLOAD_BYTES:
            return jsonify({"status": "error", "message": "이미지 크기가 너무 큽니다."}), 413
        with Image.open(io.BytesIO(raw)) as source:
            processed = preprocess_handwriting(source, size=IMAGE_SIZE)

        if processed.getbbox() is None:
            return jsonify({"status": "error", "message": "그림판에 글자를 먼저 써 주세요."}), 400

        tensor = transforms.ToTensor()(processed).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            with autocast_context():
                logits = model(tensor)
            probabilities = torch.softmax(logits.float(), dim=1)[0]
        k = min(3, len(classes))
        scores, indices = torch.topk(probabilities, k=k)
        top = [
            {"character": classes[int(i)], "score": round(float(s) * 100, 1)}
            for s, i in zip(scores.cpu(), indices.cpu())
        ]
        best = top[0]
        margin = (top[0]["score"] - top[1]["score"]) if len(top) > 1 else 100.0
        if best["score"] < 55 or margin < 8:
            note = "상위 후보의 점수 차이가 작습니다. 획을 더 크고 또렷하게 써서 다시 시도해 보세요. 점수는 보정된 확률이 아닙니다."
        else:
            note = "점수는 모델의 softmax 출력이며 실제 인식 정확도나 보정된 확률과 동일하지 않습니다."

        return jsonify({
            "status": "success",
            "prediction": best["character"],
            "confidence": best["score"],
            "top_predictions": top,
            "margin": round(margin, 1),
            "note": note,
            "model_metrics": metrics,
        })

    except FileNotFoundError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    except (ValueError, UnidentifiedImageError, OSError) as exc:
        return jsonify({"status": "error", "message": f"이미지를 처리할 수 없습니다: {exc}"}), 400
    except Exception as exc:
        print(f"인식 오류: {exc}")
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"status": "error", "message": "요청 데이터가 너무 큽니다."}), 413


if __name__ == "__main__":
    print("=" * 60)
    print("한글 손글씨 인식 AI - 공간 정보 보존 CNN")
    print("주소: http://127.0.0.1:5000")
    print(f"데이터 폴더: {DATA_DIR}")
    print(f"입력 크기: {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"장치: {DEVICE}")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
