# ⚖️ Law-Action-Assistant: 상황 맞춤형 법률 AI 에이전트

> **"법은 복잡하지만, 당신의 대처는 명확해야 합니다."**
> 대한민국 현행법령 데이터를 정밀 구조화하고, **LangGraph**의 논리 추론과 **LangSmith**의 검증 파이프라인을 통해 사용자의 상황에 최적화된 **'실행 가능한 액션 플랜(Action Plan)'**을 제공하는 차세대 법률 지원 서비스입니다.

---

## 🚀 Project Overview
단순히 법조문을 나열하는 기존 챗봇의 한계를 넘어, 본 프로젝트는 사용자가 처한 복잡한 시나리오를 분석하여 **실질적인 해결책**을 제시하는 데 집중합니다. 
로컬 환경에서 대한민국 법령 전수 수집을 성공적으로 완료하였으며, 이를 기반으로 **Kaggle**의 GPU 자원을 활용하여 고도화된 RAG 에이전트를 구축합니다.

## ✨ Key Features (Advanced Roadmap)

사용자가 자신의 상황을 입력하면, AI 에이전트가 다음과 같은 단계를 거쳐 종합적인 솔루션을 제공합니다.

1.  **🔍 상황 분석 및 법령 매칭:** 입력된 시나리오를 분석하여 조·항·호 단위로 구조화된 데이터베이스에서 관련 법령을 즉시 도출합니다.
2.  **🛡️ 연관 법률 종합 진단 (Multi-Domain):** 단일 법령뿐만 아니라, 하나의 사건에서 발생할 수 있는 **민사적 책임**과 **형사적 책임**을 동시에 진단하여 복합적인 법률 조언을 제공합니다. (**LangGraph Loop** 활용)
3.  **🎬 실행 가능한 대처법 (Action Plan) 제시:** 승소를 위해 필요한 구체적인 **증거물 체크리스트**와 단계별 대응 가이드를 생성합니다.
4.  **📑 실제 판례 사례 연결:** 사용자의 상황과 유사한 **실제 판례**를 매칭하여 답변의 신뢰성을 높이고 현실적인 감각을 제공합니다.
5.  **📝 법률 문서 초안 자동 생성:** 고소장, 내용증명 등 필수 법률 문서의 기초 드래프트(Draft)를 자동으로 작성하여 사용자의 부담을 덜어줍니다.

## 📊 Data Engineering (Status: Completed)
- **Source:** 국가법령정보센터(law.go.kr) API
- **Format:** Hierarchical JSON (조 > 항 > 호 > 목 계층 구조 유지)
- **Progress:** - [x] API 연동 및 고성능 배치 수집 엔진 구현
  - [x] 대한민국 현행법령 전수 수집 완료 (약 5,000건+)
  - [x] 개별 배치 파일(Batch 1-56) 통합 및 데이터 정제 완료
  - [x] 최종 확보 데이터: 5,567건 (JSON)
  - [ ] Kaggle Dataset 업로드 및 공개 (예정)

## 🛠 Tech Stack
- **Language:** Python 3.13 (Data Collection & Processing)
- **AI Frameworks:**LangGraph** (Implementation Planned)
- **RAG Engine:** **Retrieval-Augmented Generation** (Context-Aware AI)
- **Monitoring:** **LangSmith** (Evaluation Planned)
- **Database:** Vector DB (**FAISS** / **ChromaDB** 예정)
- **Embedding:** OpenAI Text-Embedding-3-Small (또는 HuggingFace 모델 예정)
