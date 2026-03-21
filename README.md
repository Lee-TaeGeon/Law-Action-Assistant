⚖️ Law-Action-Assistant: 상황 맞춤형 법률 AI 에이전트
"법은 복잡하지만, 당신의 대처는 명확해야 합니다."
대한민국 현행법령 데이터를 정밀 구조화하고, LangGraph의 논리 추론과 LangSmith의 검증 파이프라인을 통해 사용자의 상황에 최적화된 **'실행 가능한 액션 플랜(Action Plan)'**을 제공하는 차세대 법률 지원 서비스입니다.

🚀 Project Overview
단순히 법조문을 나열하는 기존 챗봇의 한계를 넘어, 본 프로젝트는 사용자가 처한 복잡한 시나리오를 분석하여 실질적인 해결책을 제시하는 데 집중합니다.
로컬 환경에서 대한민국 법령 전수 수집을 성공적으로 완료하였으며, 이를 기반으로 고도화된 RAG(Retrieval-Augmented Generation) 에이전트를 구축합니다.

📊 Dataset on Kaggle
본 프로젝트에서 직접 수집하고 정제한 데이터셋은 법률 AI 생태계 발전을 위해 Kaggle에 공개되어 있습니다.

🔗 South Korea Current Statutes Dataset (JSON)

Total Laws: 5,567건 (대한민국 현행법령 전수)

Total Articles: 약 220,000개 이상의 조문

Format: Hierarchical JSON (조 > 항 > 호 > 목 계층 구조 유지)

✨ Key Features (Advanced Roadmap)
사용자가 상황을 입력하면, AI 에이전트가 다음과 같은 단계를 거쳐 종합 솔루션을 제공합니다.

🔍 고정밀 법령 매칭: 조·항·호·목 단위로 정밀 구조화된 DB에서 관련 법령을 즉시 도출합니다.

🛡️ 연관 법률 종합 진단 (Multi-Domain): 사건의 이면에 숨겨진 민사적 책임과 형사적 책임을 동시에 분석합니다. (LangGraph Loop 활용)

🎬 실행 가능한 액션 플랜 제시: 법적 대응을 위해 필요한 증거물 체크리스트와 단계별 가이드를 생성합니다.

📑 실제 판례 사례 연결: 유사한 실제 판례를 매칭하여 답변의 법적 신뢰성을 확보합니다.

📝 법률 문서 초안 자동 생성: 고소장, 내용증명 등 필수 문서의 기초 드래프트(Draft)를 작성합니다.

📈 Data Engineering & RAG Pipeline (Status: Completed)
대한민국 법령의 특수성을 고려하여 검색 정확도를 극대화하는 데이터 파이프라인을 구축했습니다.

[x] 고성능 배치 수집 엔진: 국가법령정보센터 API를 통해 현행법령 5,567건 전수 수집 완료.

[x] 조문 단위 고정밀 파싱 (Hierarchical Parsing):

단순 텍스트 추출이 아닌 조(Article) > 항(Paragraph) > 호(Item) > 목(Sub-item) 위계질서 완벽 보존.

└─[호], └─[목] 시각적 인덱싱을 도입하여 RAG 모델의 문맥 이해도 향상.

[x] 대용량 Vector DB 구축 (ChromaDB):

22만 개 조문을 1,000개 단위 배치 임베딩으로 처리하여 메모리 효율성 및 안정성 확보.

ko-sroberta-multitask 모델을 활용한 고성능 한국어 문장 임베딩 적용.

🔍 데이터 구조화 예시 (Data Structure Example)
Plaintext
<근로기준법>
제17조 (근로조건의 명시)
  [항] ① 사용자는 근로계약을 체결할 때에 근로자에게 다음 각 호의 사항을 명시하여야 한다.
    └─[호] 1. 임금
    └─[호] 2. 소정근로시간
    └─[호] 3. 제55조에 따른 휴일
    └─[호] 4. 제60조에 따른 연차 유급휴가

    
🛠 Tech Stack
Language: Python 3.13

AI Frameworks: LangChain, LangGraph

Monitoring & Eval: LangSmith

Vector DB: ChromaDB (Persistent Local Storage)

Embedding: jhgan/ko-sroberta-multitask (Sentence-Transformers)

Library: langchain-huggingface, tqdm, json