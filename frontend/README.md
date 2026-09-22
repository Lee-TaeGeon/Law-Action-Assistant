# ⚖️ Law Action Assistant - Frontend

Law Action Assistant의 React 기반 Frontend입니다.

사용자가 자연어로 법률 질문을 입력하면 Railway에 배포된 FastAPI Backend와 통신하여 AI가 생성한 답변과 관련 법령 출처를 화면에 표시합니다.

---

## 🌐 Live Demo

https://law-action-assistant.vercel.app

---

# 🖥 Overview

Frontend는 기존 Streamlit 기반 UI를 React + Vite 구조로 리팩토링했습니다.

주요 역할:

- 법률 AI 상담 Chat UI
- FastAPI Backend 연동
- AI 답변 표시
- 관련 법령 Source 표시
- 대화 Session 관리
- LocalStorage 기반 대화 기록
- Sidebar 대화 목록
- Responsive Web UI

---

# 🛠 Tech Stack

| Category   | Technology        |
| ---------- | ----------------- |
| Framework  | React             |
| Build Tool | Vite              |
| Language   | JavaScript        |
| API        | Fetch / REST API  |
| Storage    | LocalStorage      |
| Deployment | Vercel            |
| Backend    | FastAPI / Railway |

---

# 🔌 Backend API

Production Backend:

```text
https://law-action-assistant-production.up.railway.app
```

Swagger:

```text
https://law-action-assistant-production.up.railway.app/docs
```

Frontend에서는 다음 API를 사용합니다.

```http
POST /api/chat
```

Request:

```json
{
  "question": "회사에서 임금을 두 달째 받지 못했습니다. 어떻게 해야 하나요?",
  "session_id": "default"
}
```

Response:

```json
{
  "category": "노동",
  "answer": "...",
  "sources": [
    {
      "law_name": "근로기준법",
      "article": "제43조",
      "content": "..."
    }
  ]
}
```

---

# 📁 Frontend Structure

```text
frontend/
│
├── src/
│   ├── components/
│   ├── services/
│   │   └── chatApi.js
│   ├── App.jsx
│   └── main.jsx
│
├── public/
├── package.json
├── vite.config.js
└── README.md
```

프로젝트 구조는 리팩토링 과정에 따라 변경될 수 있습니다.

---

# ⚙️ API Configuration

API 주소는 환경변수로 관리합니다.

현재 `chatApi.js`는 다음 값을 사용합니다.

```javascript
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
```

따라서 로컬 환경과 배포 환경에서 각각 다른 Backend를 사용할 수 있습니다.

---

# 💻 Local Development

## 1. Install

```powershell
cd frontend

npm install
```

---

## 2. Environment Variable

`frontend/.env` 파일을 생성합니다.

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Railway Backend를 직접 사용하려면:

```env
VITE_API_BASE_URL=https://law-action-assistant-production.up.railway.app
```

---

## 3. Development Server

```powershell
npm run dev
```

기본 주소:

```text
http://localhost:5173
```

---

# 🏗 Production Build

```powershell
npm run build
```

빌드 결과는 다음 폴더에 생성됩니다.

```text
dist/
```

로컬에서 Production Build를 확인하려면:

```powershell
npm run preview
```

---

# ▲ Vercel Deployment

Frontend는 Vercel에 배포되어 있습니다.

Production URL:

```text
https://law-action-assistant.vercel.app
```

Vercel Project 설정:

```text
Framework Preset
Vite

Root Directory
frontend

Build Command
npm run build

Output Directory
dist
```

Environment Variable:

```env
VITE_API_BASE_URL=https://law-action-assistant-production.up.railway.app
```

---

# 🔄 Deployment Flow

GitHub Repository의 `main` 브랜치와 Vercel을 연결해 두면 Frontend 수정 후 Push 시 자동으로 재배포됩니다.

```text
Frontend 코드 수정
       ↓
npm run build
       ↓
Git Commit
       ↓
Git Push
       ↓
GitHub main
       ↓
Vercel Auto Deploy
       ↓
Production
```

일반적인 배포 과정:

```powershell
git status

git add frontend

git commit -m "개선: 프론트엔드 UI 업데이트"

git push origin main
```

---

# 🌐 CORS

Frontend Production Origin:

```text
https://law-action-assistant.vercel.app
```

Railway Backend에서는 다음 환경변수를 통해 해당 Origin을 허용합니다.

```env
CORS_ORIGINS=https://law-action-assistant.vercel.app
```

만약 브라우저에서 API 호출 시 CORS 오류가 발생한다면 Railway의 `CORS_ORIGINS` 설정을 확인합니다.

---

# 🔍 Troubleshooting

## Backend 연결 오류

Vercel Environment Variable 확인:

```text
VITE_API_BASE_URL
```

현재 Production 값:

```text
https://law-action-assistant-production.up.railway.app
```

---

## Local API 연결 오류

FastAPI가 실행 중인지 확인합니다.

```text
http://127.0.0.1:8000/health
```

정상 응답:

```json
{
  "status": "ok"
}
```

---

## Vercel 수정사항이 반영되지 않는 경우

먼저 GitHub에 최신 Commit이 올라갔는지 확인합니다.

```powershell
git status
git log -1 --oneline
```

그다음 Vercel Deployment 상태가 `Ready`인지 확인합니다.

---

# ⚠️ Disclaimer

Law Action Assistant에서 제공하는 답변은 참고용 법률 정보이며 변호사 등 법률 전문가의 자문을 대체하지 않습니다.

실제 사건에 대한 법적 대응이 필요한 경우 전문가의 검토가 필요할 수 있습니다.

---

# 🔗 Main Project

전체 Architecture, AI Workflow, RAG, Backend 및 Railway 배포 방법은 Repository Root의 README를 참고하세요.

```text
../README.md
```

---

# 👨‍💻 Developer

**이태건**

AI Service / Backend Developer

GitHub:

https://github.com/Lee-TaeGeon
