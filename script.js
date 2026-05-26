// --- DATA MODELS ---

const RESUME_PDF = 'Adivaseresume.pdf';
const CONTACT_EMAIL = 'divaseadarsh608@gmail.com';
const FORMSUBMIT_ENDPOINT = `https://formsubmit.co/ajax/${CONTACT_EMAIL}`;

const experiences = [
    {
        period: 'Sep 2025 - Present',
        role: 'AI Full Stack Developer',
        company: 'Allwave AV Systems Pvt Ltd',
        location: 'Mumbai, Maharashtra',
        highlights: [
            'AI Full Stack Developer shipping <strong>6 production platforms</strong>—TypeScript/React, Python/FastAPI, AWS Amplify Gen 2, App Runner, Cognito, DynamoDB, SES, Lambda, Gemini, LangChain RAG, and AVIXA engineering tooling.',
            'Own architecture through deployment: CDK backends, ECR/CodeBuild CI/CD, GraphQL/AppSync schemas, multi-tenant RBAC, secrets rotation, CloudWatch/CloudTrail debugging, and India compliance (GST, MSME, Tally XML).',
            'Full technical breakdown of each system is in <a href="#production-systems" class="text-cyan-400 hover:text-cyan-300 font-semibold">Production Systems</a> below—architecture, AWS infra, and APIs per product.'
        ]
    },
    {
        period: 'Dec 2024 - Present',
        role: 'Python Backend Developer Intern',
        company: 'Aceminds Digital Pvt Ltd',
        location: 'Pune, Maharashtra',
        highlights: [
            'Designed and maintained high-performance API endpoints using <strong>FastAPI</strong> and <strong>Flask</strong>, integrating machine learning and deep learning pipelines into production services.',
            'Collaborated with front-end engineers to deploy responsive, production-ready interfaces connected to model inference backends.',
            'Architected scalable database schemas and optimized complex queries, achieving a <strong>15% reduction in data retrieval times</strong>.'
        ]
    }
];

const workProjects = [
    {
        id: 'hiro',
        title: 'HiRo',
        subtitle: 'AI HR Platform — Multi-Tenant Recruitment, Onboarding & Email Automation',
        company: 'Allwave AV Systems',
        period: 'Sep 2025 – Present',
        liveUrl: null,
        description: 'Production multi-tenant HR platform: recruitment pipelines, employee onboarding, workflow/standalone email automation, AI chatbot (Gemini), and per-user corporate email—fully deployed on AWS with GitHub → CodeBuild → ECR → App Runner CI/CD.',
        tags: ['Python', 'FastAPI', 'TypeScript', 'App Runner', 'Cognito', 'SES', 'DynamoDB', 'Gemini', 'CodeBuild', 'ECR'],
        stack: 'Python 3 · FastAPI · AWS App Runner · Amazon Cognito · SES · DynamoDB · Google Gemini · CodeBuild · ECR · CloudWatch · AWS CLI',
        architecture: [
            'Multi-tenant data model with DynamoDB-backed email templates, activity logs, and HR workflow state.',
            'FastAPI services behind App Runner with health checks, env-based config, and containerized deploys from ECR.',
            'Split automation engine: workflow-triggered sequences vs standalone campaign sends with template preview.',
            'Gemini-powered HR chatbot integrated into compose/send flows with configurable API keys and model routing.'
        ],
        infrastructure: [
            'CI/CD: GitHub → AWS CodeBuild → ECR image push → App Runner service update.',
            'Cognito user pools for auth; SES domain verification for production deliverability (corporate domains).',
            'CloudWatch logs + curl/AWS CLI runbooks for production debugging and template-sync incidents.'
        ],
        apis: [
            'REST APIs for recruitment, onboarding, email compose/send, template CRUD, and chatbot inference.',
            'DynamoDB sync endpoints for template versioning and UTC-normalized activity timestamps.'
        ],
        highlights: [
            '7+ branded HTML email templates; per-user SES sending after domain verification.',
            'Fixed production blockers: DynamoDB template sync, compose UI, preview rendering, Gemini config.',
            'Implemented workflow vs standalone automation paths with auditable send logs.',
            'Production hardening: API key rotation discipline, env secrets outside Git, App Runner redeploys.'
        ],
        impact: [
            'Centralized HR operations for hiring, onboarding, and automated candidate communication.',
            'Enterprise-grade email deliverability with verified domains and repeatable CI/CD releases.'
        ]
    },
    {
        id: 'nexo',
        title: 'Nexo',
        subtitle: 'AV Programmer Assistant — BOQ Parse, Q&A & Programming Guides',
        company: 'Allwave AV Systems · Programming AI repo',
        period: 'Sep 2025 – Present',
        liveUrl: 'https://nexo.allwaveav.com',
        description: 'TypeScript/React production app for AV programmers (Crestron, Extron, CUE, AMX, QSC, Biamp, Control4). Hybrid structured form + free-text → Gemini → code, troubleshooting docs, and full BOQ-guided programming guides. Deployed on AWS App Runner with Cognito invite-only access.',
        tags: ['TypeScript', 'React', 'Vite', 'Node.js', 'App Runner', 'Cognito', 'DynamoDB', 'S3', 'Gemini', 'XLSX'],
        stack: 'TypeScript · React 18 · Vite · Tailwind · Node.js HTTP API · @google/genai · AWS App Runner · Cognito · DynamoDB · S3 · mammoth · xlsx',
        architecture: [
            'Dual UX: Quick Ask (generate / troubleshoot / document modes) + BOQ Guided Flow (parse → questions → guide).',
            'BoqGuidedFlow: Excel/Word/text import → /api/boq/parse → line-item validation → /api/boq/questions → /api/boq/guide.',
            'Structured prompts for platform, task type, site/network/audio/video/lighting handover fields.',
            'Team snippets + per-user history stored in DynamoDB (PK userId, SK timestamp); admin provisioning UI.'
        ],
        infrastructure: [
            'AWS App Runner service with ECR images, env secrets for GEMINI_API_KEY, health endpoint /api/health.',
            'Cognito invite-only pools; ADMIN group RBAC restored via CLI + admin UI after production incidents.',
            'Lambda-ready backend modules: backend/ask, history, snippets with API Gateway authorizer pattern.',
            'Incident response: traced key exposure (App Runner env, local artifacts), rotated keys, confirmed clean Git.'
        ],
        apis: [
            'POST /api/ask — Gemini inference with mode-specific system prompts (generate, troubleshoot, document).',
            'POST /api/boq/parse · /api/boq/questions · /api/boq/guide — multi-step BOQ programming pipeline.',
            'GET/DELETE /api/history · GET/POST /api/snippets — persisted programmer session data.'
        ],
        highlights: [
            'react-markdown + syntax highlighter for deployment-ready control code output.',
            'BOQ parsers handle .xlsx/.xls/.docx with column detection (qty, rate, particulars, item code).',
            'Debugged 502 on /api/boq/questions: CloudTrail + Envoy headers → App Runner 120s vs long Gemini jobs.',
            'Secrets Manager / env-only API keys; no client-side Gemini exposure in production builds.'
        ],
        impact: [
            'Live at nexo.allwaveav.com — reduces BOQ-to-code cycle time for Allwave programming teams.',
            'Single pane for Crestron/Extron/QSC-style deliverables with auditable history and team snippets.'
        ]
    },
    {
        id: 'bingo',
        title: 'BINGO',
        subtitle: 'Bill of Quantities Intelligent Next-Gen Optimizer v2.1',
        company: 'Allwave AV Systems · BINGO-main repo',
        period: 'Sep 2025 – Present',
        liveUrl: null,
        description: 'AVIXA CTS-D compliant AI BOQ platform: questionnaire-driven generation, 2000+ curated AV products, Ask BINGO RAG assistant, brand enforcement, schematic/CAD export (DXF/Stardraw), and scheduled SES reporting—built on AWS Amplify Gen 2 + TypeScript.',
        tags: ['TypeScript', 'React', 'Amplify Gen 2', 'CDK', 'DynamoDB', 'Cognito', 'Gemini', 'RAG', 'SES', 'DXF'],
        stack: 'TypeScript 5 · React 18 · Vite · AWS Amplify Gen 2 · aws-cdk-lib · DynamoDB · AppSync GraphQL · Cognito · Lambda · SES · Gemini 2.5 Pro · Zod · ExcelJS · jsPDF',
        architecture: [
            'Amplify Gen 2 backend: auth, GraphQL data models, 8+ Lambda functions (gemini-proxy, job starter/worker/status, weekly-report, send-proposal, bootstrap-admin, user-profile, sync-user-access).',
            'Ask BINGO RAG (ragService.ts): vector + legacy DB search, IndexedDB cache (bingo-rag-cache), exact-model matching, web fallback.',
            'geminiService.ts: AV consultant system prompts, BOQ refinement, product DB grounding, size/category query routing.',
            'AVIXA engines: DMD viewing distance, ACE audio coverage, VIP camera FOV; brand prefs per component class.',
            'CAD pipeline: signal flow, rack elevations, Stardraw symbols, DXF export with 27+ layers and cable/equipment schedules.'
        ],
        infrastructure: [
            'EventBridge cron: weekly SES activity reports (Friday 13:00 UTC) to it@allwaveav.com, projects@allwaveav.com.',
            'IAM-scoped SES send permissions; DynamoDB ActivityLog scans for audit analytics.',
            'Cognito groups + bootstrap-admin Lambda for secure admin onboarding (no public API keys on client).',
            'Async Gemini job queue pattern to avoid API Gateway timeouts on large BOQ generations.'
        ],
        apis: [
            'AppSync GraphQL for products, BOQs, users, activity logs, proposals.',
            'Lambda-backed Gemini proxy and long-running job status polling endpoints.',
            'Client services: productService, activityLogService, userManagementService, askBingoService.'
        ],
        highlights: [
            'Database-first sourcing with web price estimation and automatic equipment dimension retrieval.',
            'Natural-language BOQ edits (“add 2 more speakers”) with strict brand enforcement per category.',
            'ExcelJS + jsPDF + docx export paths for client-ready proposals.',
            'deploy:secure PowerShell script + Cognito group sync automation (sync-cognito-groups.mjs).'
        ],
        impact: [
            'Transforms presales BOQ work from multi-day manual effort to AVIXA-validated, export-ready packages.',
            'Enterprise audit trail and RBAC for global integrator deployments.'
        ]
    },
    {
        id: 'pronto',
        title: 'PRONTO',
        subtitle: 'Allwave Support Brain — RAG, Jira, Voice & Ticket Automation',
        company: 'Allwave AV Systems · Support-Chatbot-main repo',
        period: 'Sep 2025 – Present',
        liveUrl: 'https://pronto.allwaveav.com',
        description: 'Enterprise FastAPI + React support platform: FAISS RAG over internal KB, LangChain + Gemini chains, live Jira ticket intelligence, Hindi/English voice I/O, image/schematic analysis, admin analytics, JWT approval workflow, and one-click AI solutions posted back to Jira.',
        tags: ['Python', 'FastAPI', 'TypeScript', 'React', 'FAISS', 'LangChain', 'Gemini', 'Jira', 'Lambda', 'boto3'],
        stack: 'Python 3.10+ · FastAPI · Uvicorn · LangChain · langchain-google-genai · FAISS · React · Vite · Gemini · Jira REST · boto3 · PyJWT · bcrypt · AWS Lambda · Amplify',
        architecture: [
            'Lazy-loaded RAGEngine + JiraService + EscalationService to survive Lambda cold-start INIT limits.',
            'FAISS vector index (local + S3 sync path) with document ingestion from data/ knowledge base folder.',
            'SupportWorkflow state machine: smart chat, ticket analysis, apply-solution, feedback loops.',
            'ADF (Atlassian Document Format) parser for rich Jira comment extraction and AI summary generation.',
            'Frontend: React + axios + react-markdown/remark-gfm; admin dashboards, ticket pagination, real-time alerts.'
        ],
        infrastructure: [
            'AWS Lambda handler (lambda_handler.py) + API Gateway; FAISS index hydration from S3 on cold start.',
            'Amplify-hosted frontend with VITE_API_URL → API Gateway prod stage; CORS for pronto.allwaveav.com.',
            'JWT auth router with admin approval gate; bcrypt password hashing; multi-device login via host IP proxy pattern.',
            'BackgroundTasks for long Jira comment posts and streaming responses where applicable.'
        ],
        apis: [
            'Smart chat, image chat, ticket chat, escalation, feedback, analytics overview endpoints.',
            'Jira: paginated ticket fetch, AI analysis comments prefixed [PRONTO - AI Analysis], customer send-back.',
            'KB rebuild, health checks, webhook handlers, schematic generation routes.',
            'Voice input/output pipelines for Hindi and English field support.'
        ],
        highlights: [
            'LangChain ChatPromptTemplate chains with Gemini for AV-domain system instructions.',
            'Browser notifications for high-priority tickets; conversation persistence with search.',
            'Production debugging across Lambda INIT timeouts, S3 KB sync, and cross-laptop dev networking.',
            'Email service templates for password reset and onboarding (PRONTO-branded HTML).'
        ],
        impact: [
            'Live at pronto.allwaveav.com — accelerates L1/L2 AV support with grounded, Jira-integrated AI.',
            'Reduces mean-time-to-resolution via automated ticket commentary and RAG-cited answers.'
        ]
    },
    {
        id: 'av-inventory',
        title: 'AV Inventory Ops',
        subtitle: 'Administrative AI — GST / MSME / Tally Compliance & 36-Table Ops Backend',
        company: 'Allwave AV Systems · Admisntrative AI repo',
        period: 'Sep 2025 – Present',
        liveUrl: null,
        description: 'Version 5.0 AV integration inventory & operations platform (ap-south-1): Amplify Gen 2 CDK backend with 36 DynamoDB tables, 19 Lambda functions, AppSync GraphQL, OpenSearch, WAF, 22 SES templates, and React 18 TypeScript PWA frontend with Gemini floating assistant.',
        tags: ['TypeScript', 'Amplify Gen 2', 'GraphQL', 'DynamoDB', 'Lambda', 'CDK', 'SES', 'OpenSearch', 'Playwright'],
        stack: 'TypeScript 5.7 · Amplify Gen 2 · aws-cdk-lib · AppSync · DynamoDB (36 tables) · 19× Lambda · Cognito · S3 · SES/SESv2 · SNS · EventBridge · Scheduler · Secrets Manager · OpenSearch · React PWA · TanStack Query',
        architecture: [
            'GraphQL schema: ProductMaster, UnitRecord (8 GSIs), GRN, DeliveryChallan, invoices, POs, BOQUpload, AMC, ServiceTicket, vendors, clients, projects, HSN DB, audit logs, chat sessions, FY counters, etc.',
            'Lambdas: alert-engine, reminder-dispatcher, invoice-scheduler, payment-reminder-sender, msme-compliance-checker, boq-parser, chatbot-handler, tally-export-generator, forex-rate-fetcher, depreciation-engine, hsn-validator, fy-rollover, tds-auto-creator, warranty-alert-monthly, amc-renewal-checker, daily-digest, client-portal-handler, user-admin.',
            'shared/ TS utilities: FY logic, GSTIN regex, HSN codes, invoice numbering, Tally XML generation.',
            'Frontend: shadcn/Radix UI, react-hook-form + Zod, TanStack Table/Virtual, Recharts, html5-qrcode, Cmd+K palette, session idle monitor, Playwright E2E.'
        ],
        infrastructure: [
            'All tables: PITR + encryption at rest + deletion protection; WAF on public endpoints.',
            'Secrets Manager for Gemini + ExchangeRate-API keys; SES production access with DKIM/SPF/DMARC.',
            'Scheduled jobs: FY rollover (Apr 1 IST), monthly TDS, warranty alerts, AMC renewal, daily digest.',
            'Vitest unit tests (backend) + Playwright mock/live E2E modes on frontend.'
        ],
        apis: [
            'AppSync GraphQL CRUD across inventory, billing, procurement, compliance, and portal modules.',
            'On-demand Lambda invocations documented in Postman collection (av-inventory.postman_collection.json).',
            'Client portal token handlers and Tally XML export generators for finance handoff.'
        ],
        highlights: [
            'India compliance: MSMED Act 2006, GSTIN validation, e-Way Bill alignment, Udyam certificate storage on S3.',
            'BOQ parser Lambda + floating Gemini chatbot widget on every major screen.',
            '22 SES HTML templates built via scripts/build-ses-templates.ts; nodemailer fallbacks.',
            'Zustand + TanStack Query for performant tables across thousands of unit records.'
        ],
        impact: [
            'Replaces fragmented spreadsheets with a single auditable ops system for Allwave back-office teams.',
            'Finance-ready exports and automated compliance schedulers reduce manual GST/MSME risk.'
        ]
    },
    {
        id: 'ez-configurator',
        title: 'EZ Configurator',
        subtitle: 'ALLWAVE Omni-Configurator — 3D AVIXA Physics & Gemini Auto-Design API',
        company: 'Allwave AV Systems · alwave-configurator repo',
        period: 'Sep 2025 – Present',
        liveUrl: null,
        description: 'Professional AV room design studio: React-Three-Fiber 3D engine with real acoustic/display physics, 42-SKU catalog (14 brands), Gemini-powered /api/chat and /api/auto-design Express backend, and PDF handover packs with GST BOQ.',
        tags: ['JavaScript', 'React', 'Three.js', 'R3F', 'Vite', 'Express', 'Gemini', 'AVIXA', 'Zustand'],
        stack: 'React 18 · Vite · Three.js · @react-three/fiber · @react-three/drei · postprocessing · Zustand · Tailwind · Express · @google/generative-ai · jsPDF · html2canvas · Framer Motion',
        architecture: [
            'engines/physics.js: DISCAS viewing distance, SPL Lp = Lw − 20log₁₀(r) − 11, Sabine RT60, PAG/NAG feedback, cable distance limits (HDMI, HDBaseT, USB, Dante, AVoIP).',
            'store/useStore.js global state; data/products.js 42 SKUs; catalog build/validate scripts for classification QA.',
            '3D scene: PBR wall materials (6 absorption coefficients), snap-to-surface device placement, SSAO/Bloom/Vignette.',
            'server/index.mjs: secure Gemini key storage, POST /api/chat, POST /api/auto-design for AI room proposals.'
        ],
        infrastructure: [
            'Split frontend (Vite dev server) + Node Express AI API (port 8787) for UAT and secure key isolation.',
            'Gemini auto-design error codes surfaced to UI; CORS-enabled local and deployed API patterns.',
            'PDF export pipeline: compliance report, BOQ with GST line items, IP addressing scheme tables.'
        ],
        apis: [
            'GET /health — service health for EZ Configurator AI sidecar.',
            'POST /api/chat — conversational AV design assistant.',
            'POST /api/auto-design — automated room layout/equipment proposals from constraints.'
        ],
        highlights: [
            'Floor legibility heatmaps (green/yellow/red) and pass/fail AVIXA badges per display.',
            'Framer Motion UI transitions; remark-gfm markdown for AI design narratives.',
            'npm run catalog:build && catalog:validate for product taxonomy integrity.',
            'Module 8 PDF handover: compliance + BOQ + network scheme in one export pack.'
        ],
        impact: [
            'Lets presales engineers validate designs with physics-backed evidence before hardware is ordered.',
            'Bridges sales visualization and engineering sign-off without leaving the browser.'
        ]
    }
];

const projects = [
    {
        title: "Algorithmic Trading & Reinforcement Learning",
        subtitle: "High-Frequency Strategy Optimization",
        description: "Implemented a Deep Q-Network (DQN) agent to automate trade execution in simulated forex markets. Designed custom rewards to penalize drawdowns and integrated live transaction fees. Achieved a backtested 12% Sharpe ratio improvement over basic momentum baselines.",
        tags: ["Reinforcement Learning", "Python", "Backtesting", "Fintech"],
        github: "https://github.com/adarshdivase",
        details: {
            problem: "Retail automated trading algorithms rely on hand-crafted heuristic rules (like MACD or RSI cross-overs). These rule-based systems struggle to adapt dynamically to regime shifts, leading to high drawdowns during volatile market transitions.",
            solution: "Designed and built an end-to-end Deep Reinforcement Learning pipeline featuring a Double Deep Q-Network (DDQN) agent in PyTorch. The environment represents price series as spatial patterns using 1D convolutional feature extractors.",
            architecture: [
                "Custom OpenAI Gym environment supporting real-time transaction fee modeling, slippage, and spread latency.",
                "Neural Network: Dual-headed (Dueling) 1D CNN + LSTM architecture to extract both high-frequency localized movements and long-term macro trends.",
                "Prioritized Experience Replay (PER) to prioritize learning from tail-risk events and high-reward trade windows.",
                "Epsilon-greedy exploration schedule decayed exponentially based on real-time rolling Sharpe ratios."
            ],
            results: [
                "Achieved an annualized return of 18.4% on out-of-sample backtests with a max drawdown of just 6.2%.",
                "Demonstrated a 12% improvement in Sharpe Ratio over standard moving-average crossover strategies.",
                "Agent successfully learned to 'sit out' (hold cash) during flat, highly compressed range markets to avoid transaction fee bleed."
            ],
            codeSnippet: `class DQNAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNAgent, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=state_dim, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Flatten()
        )
        self.value_head = nn.Linear(32, 1)
        self.advantage_head = nn.Linear(32, action_dim)

    def forward(self, state):
        features = self.feature_extractor(state)
        values = self.value_head(features)
        advantages = self.advantage_head(features)
        return values + (advantages - advantages.mean(dim=-1, keepdim=True))`
        }
    },
    {
        title: "Predictive Maintenance System for Industrial Assets",
        subtitle: "IIoT Sensor Streaming Pipeline & Failure Forecasting",
        description: "Built an end-to-end anomaly detection pipeline for multi-axis CNC machines using LSTM autoencoders. Engineered features from raw high-frequency vibrations and thermal sensors, forecasting tool wear failures up to 24 hours in advance.",
        tags: ["Anomaly Detection", "LSTMs", "IIoT", "Kafka", "Docker"],
        github: "https://github.com/adarshdivase",
        details: {
            problem: "Unscheduled downtime in heavy manufacturing facilities costs operators thousands of dollars per hour. Traditional preventative maintenance schedules rely on static calendars rather than actual machine wear signals.",
            solution: "Created an IIoT pipeline that ingests raw vibration and temperature streams, cleans telemetry via streaming processing, and runs an LSTM Autoencoder to compute a reconstruction error (anomaly score) indicating equipment degradation.",
            architecture: [
                "Raw data streaming modeled using Apache Kafka to buffer high-frequency tri-axial accelerometer inputs.",
                "Inference Engine: LSTM Autoencoder deployed inside a lightweight Docker container, tracking reconstruction error over a sliding window.",
                "Database: InfluxDB time-series database optimized for high-volume sensor writes, integrated with Grafana alerts."
            ],
            results: [
                "Forecasted critical bearing failure events 24 hours in advance with a 92% precision score.",
                "Reduced false-alarm rates by 35% compared to static threshold alerting by implementing dynamic rolling z-scores.",
                "Maintained inference latencies below 15ms per sensor window."
            ],
            codeSnippet: `def compute_reconstruction_error(model, sequence):
    # Input sequence shape: (batch_size, sequence_length, features)
    reconstructed = model.predict(sequence)
    mse = np.mean(np.square(sequence - reconstructed), axis=(1, 2))
    return mse`
        }
    },
    {
        title: "User Churn Analytics Pipeline",
        subtitle: "SaaS Operational Optimization Engine",
        description: "Constructed an automated analytics pipeline predicting customer churn for a subscription platform. Integrated an XGBoost classification backend with SHAP interpretability layers, enabling marketing teams to trigger targeted retention campaigns.",
        tags: ["XGBoost", "SHAP", "Feature Store", "FastAPI"],
        github: "https://github.com/adarshdivase",
        details: {
            problem: "Subscription platforms struggle with high customer acquisition costs. Without pre-emptive identification of accounts displaying signs of disengagement, retention campaigns are reactively sent too late.",
            solution: "Built a production-ready batch churn prediction engine that continuously scores users based on usage telemetry, customer support ticket frequency, and payment histories.",
            architecture: [
                "Feature Store: DBT transformations on Snowflake data warehouse to build aggregated user-profile feature sets.",
                "Model: Tuned XGBoost classifier with custom hyperparameter optimizations and class-imbalance weight tunings.",
                "Interpretability: Integrated SHAP library to compute localized explanation values for each predicted customer score, showing EXACTLY why the system suspects churn."
            ],
            results: [
                "Improved customer retention rates by 8% within the first 60 days of deploying the model predictions to CRM campaigns.",
                "Model achieved an Area Under the ROC Curve (AUC-ROC) of 0.89 on validation datasets.",
                "Delivered explanation vectors allowing support reps to see specific user disengagement metrics."
            ],
            codeSnippet: `import xgboost as xgb
import shap

def train_and_explain(X_train, y_train, X_val):
    model = xgb.XGBClassifier(scale_pos_weight=9.5, max_depth=6, learning_rate=0.05)
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    return model, shap_values`
        }
    },
    {
        title: 'E-commerce Sales Forecasting & Analytics',
        subtitle: 'Prophet-Based Revenue & Inventory Intelligence',
        description: 'End-to-end retail analytics platform using Facebook Prophet for demand forecasting, seasonal decomposition, and stockout risk alerts—driving smarter replenishment and revenue planning.',
        tags: ['Prophet', 'Time Series', 'Pandas', 'Streamlit'],
        github: 'https://github.com/adarshdivase',
        details: {
            problem: 'E-commerce teams struggle to align inventory with volatile demand spikes, leading to stockouts and lost revenue.',
            solution: 'Built an automated forecasting pipeline with Prophet, holiday regressors, and interactive Streamlit dashboards for category-level sales projections.',
            architecture: [
                'ETL layer cleaning SKU-level sales, promotions, and stock history.',
                'Prophet models tuned per product family with cross-validation for horizon accuracy.',
                'Alerting module flagging SKUs at risk of stockout within a 14-day window.'
            ],
            results: [
                'Improved forecast accuracy on seasonal categories versus naive baselines.',
                'Enabled proactive replenishment decisions and reduced emergency procurement cycles.',
                'Delivered executive-ready visual summaries for merchandising stakeholders.'
            ],
            codeSnippet: `from prophet import Prophet

def forecast_series(df):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    return model.predict(future)`
        }
    },
    {
        title: 'AI-Powered Trading System with Risk Analytics',
        subtitle: 'Real-Time Signals, Risk Controls & Backtesting',
        description: 'Algorithmic trading stack combining technical indicators, position sizing rules, and drawdown guardrails—with live risk dashboards and backtested performance analytics.',
        tags: ['Fintech', 'Alpaca API', 'Risk Mgmt', 'Python'],
        github: 'https://github.com/adarshdivase',
        details: {
            problem: 'Retail trading strategies often lack systematic risk controls and reproducible evaluation across market regimes.',
            solution: 'Engineered a modular trading engine with signal generation, portfolio risk limits, and performance attribution using historical and paper-trading workflows.',
            architecture: [
                'Signal engine: MACD, RSI, Bollinger Bands, and custom momentum filters.',
                'Risk module: max position size, stop-loss, and portfolio heat caps.',
                'Backtesting harness with transaction costs and slippage modeling.'
            ],
            results: [
                'Validated strategy robustness across bull, bear, and sideways regimes.',
                'Reduced maximum drawdown versus uncontrolled signal-only baselines.',
                'Produced audit-ready trade logs and risk metric reports.'
            ],
            codeSnippet: `def position_size(equity, risk_pct, stop_distance):
    risk_capital = equity * risk_pct
    return max(1, int(risk_capital / stop_distance))`
        }
    },
    {
        title: 'AI Services Toolkit Pro',
        subtitle: 'Multi-Modal AI Assistant Platform',
        description: 'Unified toolkit delivering NLP, computer vision, speech-to-text, and text-to-speech capabilities behind a single Streamlit interface with modular service routing.',
        tags: ['LLMs', 'Computer Vision', 'NLP', 'Streamlit'],
        github: 'https://github.com/adarshdivase',
        details: {
            problem: 'Teams need one interface to experiment with multiple AI modalities without rebuilding integrations for each use case.',
            solution: 'Created a plug-in architecture wrapping Hugging Face, OpenAI, and local inference backends behind consistent API contracts and UI panels.',
            architecture: [
                'Service router dispatching requests to NLP, vision, and audio pipelines.',
                'Shared auth, logging, and configuration layer for model endpoints.',
                'Streamlit front-end with upload widgets and live response previews.'
            ],
            results: [
                'Cut prototyping time for new AI demos from days to hours.',
                'Standardized error handling and latency monitoring across services.',
                'Enabled rapid stakeholder demos for text, image, and voice workflows.'
            ],
            codeSnippet: `def route_request(task, payload):
    handlers = {"nlp": nlp_service, "vision": cv_service, "audio": stt_service}
    return handlers[task].run(payload)`
        }
    },
    {
        title: 'Hybrid Predictive Maintenance System',
        subtitle: 'Supervised ML + Reinforcement Learning for Asset Health',
        description: 'Hybrid maintenance recommender combining failure classifiers with RL-based scheduling to minimize downtime and optimize service windows for industrial equipment.',
        tags: ['Reinforcement Learning', 'Scikit-learn', 'IIoT', 'MLOps'],
        github: 'https://github.com/adarshdivase',
        details: {
            problem: 'Static maintenance calendars ignore real-time degradation signals and lead to either premature service or unexpected failures.',
            solution: 'Coupled supervised failure probability models with an RL agent that learns cost-aware maintenance policies from simulated operational states.',
            architecture: [
                'Feature pipeline from vibration, temperature, and runtime telemetry.',
                'Gradient-boosted classifier estimating failure risk scores.',
                'RL policy optimizing maintenance actions under cost and uptime constraints.'
            ],
            results: [
                'Balanced maintenance spend with improved equipment availability targets.',
                'Outperformed fixed-interval scheduling in simulation benchmarks.',
                'Generated interpretable maintenance recommendations for operations teams.'
            ],
            codeSnippet: `def maintenance_reward(uptime_gain, service_cost, failure_penalty):
    return uptime_gain - service_cost - failure_penalty`
        }
    },
    {
        title: 'Customer Churn API (MLOps Deployment)',
        subtitle: 'Production FastAPI + Model Serving',
        description: 'High-accuracy churn classifier exposed via FastAPI with Dockerized deployment, health checks, and SHAP-based explanations for CRM-ready retention workflows.',
        tags: ['FastAPI', 'Docker', 'SHAP', 'MLOps'],
        github: 'https://github.com/adarshdivase',
        details: {
            problem: 'Data science models often stall in notebooks without a reliable path to production inference and monitoring.',
            solution: 'Packaged an XGBoost churn model inside a containerized FastAPI service with versioning, input validation, and explanation endpoints.',
            architecture: [
                'REST API: /predict, /explain, and /health routes.',
                'Docker image with pinned dependencies and model artifact loading on startup.',
                'SHAP explanations returned alongside probability scores for CRM integration.'
            ],
            results: [
                'Achieved strong validation AUC with sub-100ms inference on batch requests.',
                'Enabled marketing teams to trigger campaigns with reason codes per account.',
                'Demonstrated end-to-end MLOps from training notebook to deployable API.'
            ],
            codeSnippet: `@app.post("/predict")
def predict(features: ChurnFeatures):
    vector = preprocess(features)
    prob = model.predict_proba([vector])[0][1]
    return {"churn_probability": float(prob)}`
        }
    }
];

const playgroundApps = [
    {
        title: "RAG Document Intelligence",
        subtitle: "PDF Querying Engine",
        description: "Upload a research paper, extract text embeddings using SentenceTransformers, and query the content with vector similarity search mapped to a local LLM context window.",
        demoUrl: "https://adarshdivase.github.io/rag-demo/",
        tags: ["Vector Search", "Transformers", "NLP"]
    },
    {
        title: "Anomaly Inspector",
        subtitle: "Live Telemetry Scanner",
        description: "Generate synthetic sensor telemetry, inject random spikes or drift anomalies, and test real-time sliding window reconstruction classifiers inside your browser.",
        demoUrl: "https://adarshdivase.github.io/anomaly-demo/",
        tags: ["WebML", "Time Series", "D3.js"]
    },
    {
        title: "Interactive Deep Neural Net",
        subtitle: "Visual Classifier Playground",
        description: "Configure layer weights, learning rates, and activations. Visualize live decision boundaries adjusting in real-time to fit non-linear dataset patterns.",
        demoUrl: "https://adarshdivase.github.io/neural-network-demo/",
        tags: ["Machine Learning", "Neural Networks", "Data Vis"]
    }
];

const skills = [
    {
        category: "Languages & Core Engineering",
        items: [
            { name: "Python (FastAPI, async, OOP)", level: 94 },
            { name: "TypeScript & JavaScript (ES2022+)", level: 92 },
            { name: "SQL & NoSQL Data Modeling", level: 88 },
            { name: "Git, GitHub & Code Review Workflows", level: 90 },
            { name: "REST API Design & GraphQL (AppSync)", level: 88 }
        ]
    },
    {
        category: "Frontend & UI Engineering",
        items: [
            { name: "React 18 (Hooks, Router, Context)", level: 93 },
            { name: "Vite, Tailwind CSS & Responsive UI", level: 92 },
            { name: "Three.js & React-Three-Fiber (3D)", level: 85 },
            { name: "Zustand, TanStack Query/Table", level: 86 },
            { name: "shadcn/Radix UI & react-hook-form + Zod", level: 84 }
        ]
    },
    {
        category: "AWS Cloud & Serverless",
        items: [
            { name: "AWS Amplify Gen 2 & CDK (aws-cdk-lib)", level: 90 },
            { name: "Lambda, API Gateway & App Runner", level: 91 },
            { name: "DynamoDB (GSIs, PITR, access patterns)", level: 90 },
            { name: "Cognito, IAM, SES, SNS, S3, Secrets Manager", level: 89 },
            { name: "EventBridge, CloudWatch, CloudTrail, ECR, CodeBuild", level: 87 }
        ]
    },
    {
        category: "Generative AI, RAG & LLM Ops",
        items: [
            { name: "Google Gemini API (@google/genai)", level: 93 },
            { name: "RAG, FAISS & LangChain Pipelines", level: 90 },
            { name: "Prompt Engineering & System Instructions", level: 92 },
            { name: "OpenAI API & Hugging Face Transformers", level: 85 },
            { name: "Long-running AI Jobs & Timeout Mitigation", level: 88 }
        ]
    },
    {
        category: "Backend, Integrations & DevOps",
        items: [
            { name: "FastAPI, Uvicorn, Flask & Node/Express", level: 91 },
            { name: "JWT, bcrypt & Role-Based Access Control", level: 88 },
            { name: "Jira REST & Atlassian ADF Parsing", level: 86 },
            { name: "Docker, CI/CD & Production Incident Response", level: 87 },
            { name: "Playwright E2E & Vitest Unit Testing", level: 82 }
        ]
    },
    {
        category: "Machine Learning & Data Science",
        items: [
            { name: "PyTorch, TensorFlow & Scikit-Learn", level: 90 },
            { name: "XGBoost, SHAP, LIME & Model Explainability", level: 88 },
            { name: "Reinforcement Learning (DQN, PPO)", level: 80 },
            { name: "Pandas, NumPy, Prophet & Time Series", level: 92 },
            { name: "Kafka, ETL & Feature Pipelines", level: 78 }
        ]
    },
    {
        category: "AV Domain & Enterprise Tooling",
        items: [
            { name: "AVIXA Standards (DMD, ACE, DISCAS, SPL)", level: 90 },
            { name: "BOQ/CAD/DXF & Stardraw Symbol Pipelines", level: 88 },
            { name: "Crestron, Extron, QSC, Biamp Programming Context", level: 87 },
            { name: "India GST, HSN, MSME & Tally XML Export", level: 86 },
            { name: "ExcelJS, jsPDF, mammoth, xlsx Document I/O", level: 89 }
        ]
    },
    {
        category: "Deployment & Observability",
        items: [
            { name: "Vercel, Render & Static Site Hosting", level: 84 },
            { name: "GitHub Actions & Amplify Hosting/CDN", level: 86 },
            { name: "OpenSearch & Structured Logging", level: 80 },
            { name: "WAF, CORS & Multi-Origin Production Config", level: 85 },
            { name: "Streamlit, Plotly & Recharts Dashboards", level: 88 }
        ]
    }
];

const blogPosts = [
    {
        id: "backtesting-reinforcement-learning",
        title: "Demystifying Reinforcement Learning in Live Financial Markets",
        date: "May 10, 2026",
        tag: "Reinforcement Learning",
        excerpt: "Why model-free reinforcement learning algorithms struggle in non-stationary reward landscapes, and how we can architect robust sensory abstractions to mitigate market regime drift.",
        content: `
            <div class="prose-custom">
                <p>Applying model-free reinforcement learning algorithms like Double DQN or PPO directly to raw financial price series is a recipe for rapid capital depletion. Financial markets are notorious for their non-stationary behavior, meaning the statistical properties of the price data drift over time, rendering historical patterns obsolete.</p>
                
                <h3 class="text-xl font-bold text-white mt-6 mb-3">The Challenge of Non-Stationarity</h3>
                <p>When an agent is trained in a simulated regime, it learns policy pathways designed for that specific set of market conditions (e.g. low-volatility bullish uptrend). When the market shifts into a high-volatility sideways churn, the reward outputs decay. In ML terms, the environmental state transitions (s_t -> s_{t+1}) violate the Markov property because hidden macro variables dictate these shifts.</p>
                
                <h3 class="text-xl font-bold text-white mt-6 mb-3">Architectual Strategies for Mitigation</h3>
                <p>To build RL agents that survive regime transitions, we must apply three key design patterns:</p>
                <ul>
                    <li><strong>Fourier & Wavelet Transforms:</strong> Rather than feeding raw prices or standard moving averages, decompose price windows into frequency components to capture cyclic momentum independently of scale.</li>
                    <li><strong>Dynamic Reward Rescaling:</strong> Implement rolling z-score scaling on transaction payoffs. This forces the agent's gradients to focus on relative performance rather than nominal currency values.</li>
                    <li><strong>Stochastic Regularization:</strong> Apply dropout to the recurrent LSTM network blocks to prevent the agent from over-indexing on localized sequences.</li>
                </ul>
            </div>
        `
    },
    {
        id: "lstm-autoencoders-anomaly",
        title: "Architecting LSTM Autoencoders for IIoT Anomaly Detection",
        date: "Apr 28, 2026",
        tag: "Anomaly Detection",
        excerpt: "An architectural guide to deploying autoencoders for high-frequency sensor telemetry. Discover strategies for tuning sequence windows and choosing reconstruction error thresholds.",
        content: `
            <div class="prose-custom">
                <p>Industrial assets like CNC spindles and wind turbine bearings produce high-frequency vibration streams that standard threshold alerts cannot analyze. LSTM Autoencoders offer a powerful solution: they learn the normal operating signature of a machine and flag deviations before a physical failure occurs.</p>
                
                <h3 class="text-xl font-bold text-white mt-6 mb-3">The Autoencoder Mechanism</h3>
                <p>An autoencoder is trained exclusively on normal, non-anomalous operational data. It compresses a sliding time-series window into a low-dimensional bottleneck representation (encoder) and then attempts to reconstruct the original input (decoder). Because the network has never seen failure sequences, its reconstruction error increases dramatically when abnormal wear signals are present.</p>
                
                <h3 class="text-xl font-bold text-white mt-6 mb-3">Key Hyperparameter Tradeoffs</h3>
                <ul>
                    <li><strong>Sequence Length:</strong> Too short (e.g., 10 steps) and the model cannot capture cyclic vibrations; too long (e.g., 500 steps) and the reconstruction gradients vanish, or latency rises. We found a sequence window of 64 to 128 steps represents the sweet spot for tri-axial vibration telemetry.</li>
                    <li><strong>Bottleneck Size:</strong> Restricting the bottleneck forces the network to learn generalized features rather than memorizing noise.</li>
                </ul>
            </div>
        `
    },
    {
        id: "fastapi-ml-deployment",
        title: "FastAPI Patterns for Low-Latency Machine Learning Pipelines",
        date: "Mar 15, 2026",
        tag: "Software Engineering",
        excerpt: "How to avoid common bottlenecks when serving PyTorch or TensorFlow models via FastAPI. Learn about model warming, async worker tuning, and connection pooling.",
        content: `
            <div class="prose-custom">
                <p>Deploying a model inside a FastAPI wrapper is easy, but optimizing it to handle hundreds of concurrent requests with sub-50ms latencies requires careful architectural adjustments.</p>
                
                <h3 class="text-xl font-bold text-white mt-6 mb-3">1. Eliminate Cold-Start Latency via Model Warming</h3>
                <p>Never load your PyTorch or TensorFlow model weights inside an active request route. Load the model globally during the FastAPI startup event and execute a single dummy forward pass. This compiles any internal lazy-evaluated operations before accepting live traffic.</p>
                
                <h3 class="text-xl font-bold text-white mt-6 mb-3">2. Async Event Loop Bottlenecks</h3>
                <p>Model forward passes are CPU-bound operations. If you call model inference inside an <code>async def</code> route without offloading it, you will block the FastAPI event loop, causing other requests to wait. Use standard <code>def</code> endpoints so FastAPI runs the inference in an external thread pool, or use <code>anyio.to_thread.run_sync</code> to offload computation.</p>
            </div>
        `
    }
];

// --- FORM & MODAL UTILITIES ---
async function submitPortfolioForm(formData, options = {}) {
    formData.append('_captcha', 'false');
    formData.append('_template', 'table');
    if (options.autoresponse) {
        formData.append('_autoresponse', options.autoresponse);
    }

    const response = await fetch(FORMSUBMIT_ENDPOINT, {
        method: 'POST',
        body: formData,
        headers: { Accept: 'application/json' }
    });

    let payload = {};
    try {
        payload = await response.json();
    } catch {
        payload = {};
    }

    const isSuccess = response.ok && (payload.success === true || payload.success === 'true' || response.status === 200);
    return { isSuccess, payload, status: response.status };
}

function setContactFormStatus(message, isError = false) {
    const statusEl = document.getElementById('contact-form-status');
    if (!statusEl) return;
    statusEl.textContent = message;
    statusEl.classList.remove('hidden', 'contact-status-success', 'contact-status-error');
    statusEl.classList.add(isError ? 'contact-status-error' : 'contact-status-success');
}

function renderExperienceTimeline() {
    const timeline = document.getElementById('experience-timeline');
    if (!timeline) return;

    timeline.innerHTML = '';
    experiences.forEach((exp, index) => {
        const item = document.createElement('div');
        item.className = 'relative grid md:grid-cols-2 gap-8 items-start md:even:flex-row-reverse';
        item.innerHTML = `
            <div class="absolute left-4 md:left-1/2 -translate-x-[7px] w-3.5 h-3.5 rounded-full bg-cyan-400 border-4 border-slate-950 z-20 shadow-md shadow-cyan-400/50"></div>
            <div class="pl-8 md:pl-0 ${index % 2 === 0 ? 'md:pr-12 md:text-right' : 'md:pl-12 md:order-2'}">
                <span class="text-xs font-mono text-cyan-400 font-bold bg-cyan-500/10 border border-cyan-500/25 px-2.5 py-1 rounded-full">${exp.period}</span>
                <h3 class="text-xl font-bold text-white mt-3">${exp.role}</h3>
                <h4 class="text-slate-400 text-sm font-semibold mt-1">${exp.company}, ${exp.location}</h4>
            </div>
            <div class="pl-8 ${index % 2 === 0 ? 'md:pl-12' : 'md:pr-12 md:order-1 md:text-right'}">
                <div class="card-bg p-6 md:p-8 rounded-2xl relative overflow-hidden group">
                    <div class="absolute top-0 left-0 w-full h-[2px] bg-gradient-to-r from-cyan-500 to-indigo-500"></div>
                    <ul class="space-y-3 text-slate-300 text-sm ${index % 2 === 1 ? 'md:text-right' : ''}">
                        ${exp.highlights.map(point => `<li class="flex gap-2 ${index % 2 === 1 ? 'md:flex-row-reverse' : ''}"><span class="text-cyan-400 shrink-0">✦</span><span>${point}</span></li>`).join('')}
                    </ul>
                </div>
            </div>
        `;
        timeline.appendChild(item);
    });
}

const resumeModal = document.getElementById('resume-modal');

function showResumeModal() {
    if (!resumeModal) return;
    const iframe = document.getElementById('resume-iframe');
    const downloadBtn = document.getElementById('resume-download-btn');
    if (iframe) iframe.src = RESUME_PDF;
    if (downloadBtn) downloadBtn.href = RESUME_PDF;
    resumeModal.style.display = 'flex';
    document.body.style.overflow = 'hidden';
}

function hideResumeModal() {
    if (!resumeModal) return;
    resumeModal.style.display = 'none';
    document.body.style.overflow = '';
    const iframe = document.getElementById('resume-iframe');
    if (iframe) iframe.src = 'about:blank';
}

function renderProductionSystems() {
    const grid = document.getElementById('production-systems-grid');
    if (!grid) return;

    grid.innerHTML = '';
    workProjects.forEach((proj, idx) => {
        const card = document.createElement('article');
        card.className = 'work-project-card card-bg p-6 rounded-2xl relative overflow-hidden group cursor-pointer flex flex-col justify-between h-full border border-indigo-500/10';
        card.innerHTML = `
            <div class="space-y-4 relative z-10">
                <div class="flex items-center justify-between gap-2 flex-wrap">
                    <span class="text-[10px] font-mono text-indigo-300 font-bold bg-indigo-500/15 border border-indigo-500/25 px-2 py-0.5 rounded-full uppercase tracking-wider">Production</span>
                    <span class="text-[10px] font-mono text-slate-500">${proj.period}</span>
                </div>
                <div class="space-y-1">
                    <h3 class="text-xl font-bold text-white group-hover:text-indigo-300 transition-colors">${proj.title}</h3>
                    <p class="text-xs font-mono text-indigo-400/90 font-medium leading-snug">${proj.subtitle}</p>
                </div>
                <p class="text-slate-400 text-sm leading-relaxed line-clamp-4">${proj.description}</p>
                <p class="text-[10px] font-mono text-slate-500 leading-relaxed line-clamp-2">${proj.stack}</p>
                <p class="text-[10px] font-mono text-slate-600 uppercase tracking-wider">Build_${String(idx + 1).padStart(2, '0')} · ${proj.company}</p>
            </div>
            <div class="flex flex-wrap gap-1.5 pt-5 relative z-10">
                ${proj.tags.slice(0, 6).map(tag => `<span class="tag text-[10px] font-mono text-indigo-200/90 bg-indigo-950/30 border border-indigo-800/40 px-2 py-0.5 rounded-md">${tag}</span>`).join('')}
                ${proj.tags.length > 6 ? `<span class="text-[10px] font-mono text-slate-500 px-1">+${proj.tags.length - 6}</span>` : ''}
            </div>
        `;
        card.addEventListener('click', () => showWorkProjectModal(proj));
        grid.appendChild(card);
    });
}

function showWorkProjectModal(proj) {
    if (!modal || !modalContent) return;

    const liveLink = proj.liveUrl
        ? `<a href="${proj.liveUrl}" target="_blank" rel="noopener" class="inline-flex items-center gap-2 text-xs font-mono text-cyan-400 hover:text-cyan-300 font-bold border border-cyan-500/30 px-3 py-1.5 rounded-lg">Live ↗</a>`
        : '<span class="text-xs font-mono text-slate-500 border border-slate-800 px-3 py-1.5 rounded-lg">Internal / Enterprise</span>';

    modalContent.innerHTML = `
        <button id="close-modal" class="absolute top-4 right-4 text-slate-400 hover:text-white p-2 rounded-full hover:bg-slate-900 transition-colors z-30" type="button" aria-label="Close">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
        </button>
        <div class="space-y-8 relative z-10">
            <div class="space-y-3">
                <div class="flex flex-wrap items-center gap-2">
                    <span class="text-xs font-mono text-indigo-300 font-bold bg-indigo-500/15 border border-indigo-500/25 px-2.5 py-1 rounded-full">Production System</span>
                    <span class="text-xs font-mono text-slate-500">${proj.company} · ${proj.period}</span>
                    ${liveLink}
                </div>
                <h2 class="text-2xl md:text-3xl font-black text-white">${proj.title}</h2>
                <p class="text-slate-400 font-mono text-xs md:text-sm text-indigo-400/90">${proj.subtitle}</p>
                <p class="text-slate-300 text-sm leading-relaxed">${proj.description}</p>
            </div>

            <div class="rounded-xl border border-slate-800 bg-slate-950/50 p-4">
                <h4 class="text-xs font-mono text-slate-400 uppercase tracking-wider mb-2">Full Tech Stack</h4>
                <p class="text-slate-300 text-sm leading-relaxed">${proj.stack}</p>
            </div>

            ${proj.architecture ? `
            <div class="space-y-2">
                <h4 class="text-xs font-mono text-slate-400 uppercase tracking-wider">System Architecture</h4>
                <ul class="space-y-2 text-slate-300 text-sm">
                    ${proj.architecture.map(item => `<li class="flex gap-2"><span class="text-indigo-400 shrink-0">▸</span><span>${item}</span></li>`).join('')}
                </ul>
            </div>` : ''}

            ${proj.infrastructure ? `
            <div class="space-y-2">
                <h4 class="text-xs font-mono text-slate-400 uppercase tracking-wider">AWS & Infrastructure</h4>
                <ul class="space-y-2 text-slate-300 text-sm">
                    ${proj.infrastructure.map(item => `<li class="flex gap-2"><span class="text-cyan-400 shrink-0">▸</span><span>${item}</span></li>`).join('')}
                </ul>
            </div>` : ''}

            ${proj.apis && proj.apis.length ? `
            <div class="space-y-2">
                <h4 class="text-xs font-mono text-slate-400 uppercase tracking-wider">APIs & Integrations</h4>
                <ul class="space-y-2 text-slate-300 text-sm font-mono text-[12px]">
                    ${proj.apis.map(item => `<li class="flex gap-2"><span class="text-purple-400 shrink-0">›</span><span>${item}</span></li>`).join('')}
                </ul>
            </div>` : ''}

            <div class="grid md:grid-cols-2 gap-8">
                <div class="space-y-2">
                    <h4 class="text-xs font-mono text-slate-400 uppercase tracking-wider">Engineering Highlights</h4>
                    <ul class="space-y-2 text-slate-300 text-sm">
                        ${proj.highlights.map(item => `<li class="flex gap-2"><span class="text-indigo-400 shrink-0">✦</span><span>${item}</span></li>`).join('')}
                    </ul>
                </div>
                <div class="space-y-2">
                    <h4 class="text-xs font-mono text-slate-400 uppercase tracking-wider">Business Impact</h4>
                    <ul class="space-y-2 text-slate-300 text-sm">
                        ${proj.impact.map(item => `<li class="flex gap-2"><span class="text-emerald-400 shrink-0">✓</span><span>${item}</span></li>`).join('')}
                    </ul>
                </div>
            </div>

            <div class="flex flex-wrap gap-1.5 pt-2 border-t border-slate-900">
                ${proj.tags.map(tag => `<span class="tag text-[10px] font-mono text-indigo-200/90 bg-indigo-950/30 border border-indigo-800/40 px-2 py-0.5 rounded-md">${tag}</span>`).join('')}
            </div>
        </div>
    `;

    modal.style.display = 'flex';
    document.body.style.overflow = 'hidden';

    const closeBtn = modalContent.querySelector('#close-modal');
    if (closeBtn) closeBtn.addEventListener('click', hideProjectModal);
}

// --- UI LOGIC ---
document.addEventListener('DOMContentLoaded', () => {
    // Global mouse coordinate tracker for background halo
    document.addEventListener('mousemove', (e) => {
        document.documentElement.style.setProperty('--mouse-x', `${e.clientX}px`);
        document.documentElement.style.setProperty('--mouse-y', `${e.clientY}px`);
    });

    // Mobile Menu Toggle
    const mobileMenuBtn = document.getElementById('mobile-menu-button');
    const mobileMenu = document.getElementById('mobile-menu');
    if (mobileMenuBtn && mobileMenu) {
        mobileMenuBtn.addEventListener('click', () => {
            mobileMenu.classList.toggle('hidden');
        });
    }

    renderExperienceTimeline();
    renderProductionSystems();

    const viewResumeBtn = document.getElementById('view-resume-btn');
    const closeResumeBtn = document.getElementById('close-resume-modal');
    if (viewResumeBtn) viewResumeBtn.addEventListener('click', showResumeModal);
    if (closeResumeBtn) closeResumeBtn.addEventListener('click', hideResumeModal);
    if (resumeModal) {
        resumeModal.addEventListener('click', (e) => {
            if (e.target === resumeModal) hideResumeModal();
        });
    }
    window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && resumeModal && resumeModal.style.display === 'flex') {
            hideResumeModal();
        }
    });

    // Scroll Progress bar
    window.addEventListener('scroll', () => {
        const winScroll = document.documentElement.scrollTop || document.body.scrollTop;
        const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
        const scrolled = (winScroll / height) * 100;
        const progressBar = document.getElementById('scroll-progress');
        if (progressBar) {
            progressBar.style.width = scrolled + '%';
        }
    });

    // --- RENDER SELECTED PROJECTS ---
    const projectsGrid = document.getElementById('projects-grid');
    if (projectsGrid) {
        projectsGrid.innerHTML = '';
        projects.forEach((proj, idx) => {
            const projectCard = document.createElement('div');
            projectCard.className = 'project-card card-bg p-6 rounded-2xl relative overflow-hidden group cursor-pointer flex flex-col justify-between h-full';
            projectCard.innerHTML = `
                <div class="space-y-4 relative z-10">
                    <div class="flex items-center justify-between">
                        <span class="text-xs font-mono text-cyan-400 font-bold bg-cyan-500/10 border border-cyan-500/20 px-2 py-0.5 rounded-full">Proj_0${idx + 1}</span>
                        <div class="flex gap-2">
                            <a href="${proj.github}" target="_blank" class="text-slate-400 hover:text-white transition-colors" onclick="event.stopPropagation();">
                                <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path fill-rule="evenodd" clip-rule="evenodd" d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.17 6.839 9.49.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.603-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.464-1.11-1.464-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.831.092-.646.35-1.086.636-1.336-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.579.688.481C19.138 20.167 22 16.418 22 12c0-5.523-4.477-10-10-10z"></path></svg>
                            </a>
                        </div>
                    </div>
                    <div class="space-y-1">
                        <h3 class="text-xl font-bold text-white group-hover:text-cyan-400 transition-colors">${proj.title}</h3>
                        <p class="text-xs font-mono text-indigo-400 font-medium">${proj.subtitle}</p>
                    </div>
                    <p class="text-slate-400 text-sm leading-relaxed">${proj.description}</p>
                </div>
                <div class="flex flex-wrap gap-1.5 pt-6 relative z-10">
                    ${proj.tags.map(tag => `<span class="tag text-xs font-mono text-cyan-300 bg-cyan-950/20 border border-cyan-800/30 px-2 py-0.5 rounded-md">${tag}</span>`).join('')}
                </div>
            `;
            // Add click listener to show the details modal
            projectCard.addEventListener('click', () => {
                showProjectModal(proj);
            });
            projectsGrid.appendChild(projectCard);
        });
    }

    // --- RENDER PLAYGROUND APPS ---
    const playgroundGrid = document.getElementById('playground-apps-grid');
    if (playgroundGrid) {
        playgroundGrid.innerHTML = '';
        playgroundApps.forEach((app, idx) => {
            const appCard = document.createElement('div');
            appCard.className = 'card-bg p-6 rounded-2xl relative overflow-hidden group flex flex-col justify-between h-full';
            appCard.innerHTML = `
                <div class="space-y-4 relative z-10">
                    <div class="flex items-center justify-between">
                        <span class="text-xs font-mono text-indigo-400 font-bold bg-indigo-500/10 border border-indigo-500/20 px-2 py-0.5 rounded-full">App_0${idx + 1}</span>
                        <a href="${app.demoUrl}" target="_blank" class="text-slate-400 hover:text-cyan-400 transition-colors">
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"></path></svg>
                        </a>
                    </div>
                    <div class="space-y-1">
                        <h3 class="text-lg font-bold text-white group-hover:text-cyan-400 transition-colors">${app.title}</h3>
                        <p class="text-xs font-mono text-cyan-500 font-medium">${app.subtitle}</p>
                    </div>
                    <p class="text-slate-400 text-sm leading-relaxed">${app.description}</p>
                </div>
                <div class="flex flex-wrap gap-1.5 pt-6 relative z-10">
                    ${app.tags.map(tag => `<span class="tag text-[10px] font-mono text-indigo-300 bg-indigo-950/20 border border-indigo-800/30 px-2 py-0.5 rounded-md">${tag}</span>`).join('')}
                </div>
            `;
            playgroundGrid.appendChild(appCard);
        });
    }

    // --- RENDER TECHNICAL ARSENAL (SKILLS) ---
    const skillsGrid = document.getElementById('skills');
    if (skillsGrid) {
        // Clear previous skills list (keeping section header)
        const header = skillsGrid.querySelector('h2');
        skillsGrid.innerHTML = '';
        if (header) skillsGrid.appendChild(header);
        
        skills.forEach(skillCat => {
            const skillCard = document.createElement('div');
            skillCard.className = 'card-bg p-8 rounded-2xl relative overflow-hidden group';
            skillCard.innerHTML = `
                <div class="absolute top-0 left-0 w-full h-[2px] bg-gradient-to-r from-cyan-500 to-indigo-500"></div>
                <h3 class="text-lg font-bold text-white mb-6 relative z-10">${skillCat.category}</h3>
                <div class="space-y-4 relative z-10">
                    ${skillCat.items.map(item => `
                        <div class="space-y-2">
                            <div class="flex justify-between text-xs font-mono">
                                <span class="text-slate-300 font-medium">${item.name}</span>
                                <span class="text-cyan-400">${item.level}%</span>
                            </div>
                            <div class="w-full h-1.5 bg-slate-950 rounded-full overflow-hidden">
                                <div class="h-full bg-gradient-to-r from-cyan-400 to-indigo-500 rounded-full transition-all duration-1000" style="width: ${item.level}%"></div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            `;
            skillsGrid.appendChild(skillCard);
        });
    }

    // --- RENDER BLOG CONTENT & TAG FILTERS ---
    const blogPostsContainer = document.getElementById('blog-posts-container');
    const blogFilters = document.getElementById('blog-filters');
    const recentPostsContainer = document.getElementById('recent-posts-container');
    let currentTagFilter = 'All';

    function renderBlog() {
        if (!blogPostsContainer) return;
        blogPostsContainer.innerHTML = '';
        
        const filteredPosts = currentTagFilter === 'All' 
            ? blogPosts : blogPosts.filter(post => post.tag === currentTagFilter);
        
        if (filteredPosts.length === 0) {
            blogPostsContainer.innerHTML = `<p class="text-slate-400 font-mono text-sm py-8">No articles found matching this tag.</p>`;
            return;
        }

        filteredPosts.forEach(post => {
            const postCard = document.createElement('div');
            postCard.className = 'card-bg p-8 rounded-2xl relative overflow-hidden group space-y-4';
            postCard.innerHTML = `
                <div class="absolute top-0 left-0 w-[2px] h-full bg-gradient-to-b from-cyan-500 to-indigo-500"></div>
                <div class="flex items-center justify-between text-xs font-mono">
                    <span class="text-slate-500">${post.date}</span>
                    <span class="text-cyan-400 font-bold bg-cyan-500/10 border border-cyan-500/20 px-2.5 py-0.5 rounded-full">${post.tag}</span>
                </div>
                <h3 class="text-xl font-bold text-white group-hover:text-cyan-400 transition-colors duration-300">${post.title}</h3>
                <p class="text-slate-300 text-sm leading-relaxed">${post.excerpt}</p>
                <div class="blog-full-content hidden border-t border-slate-900 pt-6 mt-6">
                    ${post.content}
                </div>
                <button class="read-more-btn inline-flex items-center gap-2 text-xs font-mono text-cyan-400 font-bold hover:text-cyan-300 transition-colors mt-2">
                    <span>Read Article</span>
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
                </button>
            `;
            
            // Wire expand/collapse click listener
            const readMoreBtn = postCard.querySelector('.read-more-btn');
            const fullContent = postCard.querySelector('.blog-full-content');
            readMoreBtn.addEventListener('click', () => {
                const isExpanded = !fullContent.classList.contains('hidden');
                if (isExpanded) {
                    fullContent.classList.add('hidden');
                    readMoreBtn.querySelector('span').innerText = 'Read Article';
                    readMoreBtn.querySelector('svg').style.transform = 'rotate(0deg)';
                } else {
                    fullContent.classList.remove('hidden');
                    readMoreBtn.querySelector('span').innerText = 'Collapse Article';
                    readMoreBtn.querySelector('svg').style.transform = 'rotate(180deg)';
                }
            });

            blogPostsContainer.appendChild(postCard);
        });
    }

    function renderBlogFilters() {
        if (!blogFilters) return;
        const allTags = ['All', ...new Set(blogPosts.map(post => post.tag))];
        blogFilters.innerHTML = '';
        allTags.forEach(tag => {
            const btn = document.createElement('button');
            btn.className = `px-4 py-1.5 rounded-full text-xs font-mono font-medium transition-all duration-300 border ${
                currentTagFilter === tag 
                ? 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30 shadow-[0_0_10px_rgba(6,182,212,0.15)]' 
                : 'bg-slate-950/40 text-slate-400 border-slate-900 hover:text-slate-200'
            }`;
            btn.innerText = tag;
            btn.addEventListener('click', () => {
                currentTagFilter = tag;
                renderBlogFilters();
                renderBlog();
            });
            blogFilters.appendChild(btn);
        });
    }

    function renderRecentPostsWidget() {
        if (!recentPostsContainer) return;
        recentPostsContainer.innerHTML = '';
        blogPosts.slice(0, 3).forEach(post => {
            const link = document.createElement('a');
            link.href = '#blog';
            link.className = 'block p-3.5 rounded-xl border border-slate-950 hover:border-slate-900/60 hover:bg-slate-950/20 group transition-all';
            link.innerHTML = `
                <span class="text-[10px] font-mono text-cyan-500 font-semibold block mb-1">${post.tag}</span>
                <span class="text-xs text-slate-300 font-medium group-hover:text-white line-clamp-1 transition-colors">${post.title}</span>
            `;
            link.addEventListener('click', () => {
                currentTagFilter = 'All';
                renderBlogFilters();
                renderBlog();
            });
            recentPostsContainer.appendChild(link);
        });
    }

    renderBlogFilters();
    renderBlog();
    renderRecentPostsWidget();

    // --- QUICK POLL INTERACTIVE COMPONENT ---
    const pollOptionsContainer = document.getElementById('poll-options');
    const pollFeedback = document.getElementById('poll-feedback');
    
    // Simulate initial poll data
    const pollVotes = {
        'Generative AI & LLMs': 124,
        'Computer Vision': 56,
        'MLOps & Deployment': 78
    };

    if (pollOptionsContainer) {
        const optionButtons = pollOptionsContainer.querySelectorAll('.poll-option');
        optionButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                const choice = btn.innerText.trim();
                pollVotes[choice] = (pollVotes[choice] || 0) + 1;
                
                // Calculate percentages
                const total = Object.values(pollVotes).reduce((a, b) => a + b, 0);
                
                // Show results inline
                pollOptionsContainer.innerHTML = '';
                Object.entries(pollVotes).forEach(([key, val]) => {
                    const pct = Math.round((val / total) * 100);
                    const isUserChoice = key === choice;
                    
                    const resultRow = document.createElement('div');
                    resultRow.className = 'space-y-2';
                    resultRow.innerHTML = `
                        <div class="flex justify-between text-xs font-mono">
                            <span class="${isUserChoice ? 'text-cyan-400 font-bold' : 'text-slate-300'}">${key}</span>
                            <span class="${isUserChoice ? 'text-cyan-400 font-bold' : 'text-slate-500'}">${pct}% (${val})</span>
                        </div>
                        <div class="w-full h-2 bg-slate-950 rounded-full overflow-hidden">
                            <div class="h-full bg-gradient-to-r ${isUserChoice ? 'from-cyan-400 to-indigo-500' : 'from-slate-800 to-slate-700'} rounded-full transition-all duration-1000" style="width: 0%"></div>
                        </div>
                    `;
                    pollOptionsContainer.appendChild(resultRow);
                    
                    // Trigger progress bar slide-in animation
                    setTimeout(() => {
                        resultRow.querySelector('.h-full').style.width = `${pct}%`;
                    }, 50);
                });
                
                if (pollFeedback) {
                    pollFeedback.classList.remove('hidden');
                    showToast('Quick Poll choice logged. Live telemetry updated.');
                }
            });
        });
    }

    // --- FORM SUBMISSIONS (TO GMAIL VIA FORMSUBMIT AJAX API) ---
    // Q&A Question form submission
    const commentForm = document.getElementById('comment-form');
    if (commentForm) {
        commentForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const name = document.getElementById('comment-name').value;
            const text = document.getElementById('comment-text').value;
            const submitBtn = commentForm.querySelector('button[type="submit"]');
            
            // Add loading spinner class/styles to submit button
            submitBtn.disabled = true;
            const originalBtnHTML = submitBtn.innerHTML;
            submitBtn.innerHTML = `
                <svg class="animate-spin -ml-1 mr-3 h-4.5 w-4.5 text-white inline-block" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span>Transmitting...</span>
            `;

            const formData = new FormData();
            formData.append('name', name);
            formData.append('message', text);
            formData.append('_subject', `Portfolio AMA Question from ${name}`);

            submitPortfolioForm(formData, {
                autoresponse: 'Thanks for your question! Adarsh will review it and get back to you soon.'
            })
            .then(({ isSuccess }) => {
                if (isSuccess) {
                    showToast('Question sent successfully! Check your inbox for a confirmation.');
                    commentForm.reset();
                } else {
                    showToast('Could not send question. Please email divaseadarsh608@gmail.com directly.');
                }
            })
            .catch(error => {
                showToast('Network error while sending your question.');
                console.error(error);
            })
            .finally(() => {
                submitBtn.disabled = false;
                submitBtn.innerHTML = originalBtnHTML;
            });
        });
    }

    // Contact Form submission
    const contactForm = document.getElementById('contact-form');
    if (contactForm) {
        contactForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const honey = document.getElementById('contact-honey');
            if (honey && honey.value.trim()) return;

            const name = document.getElementById('contact-name').value.trim();
            const email = document.getElementById('contact-email').value.trim();
            const subject = document.getElementById('contact-subject').value.trim();
            const message = document.getElementById('contact-message').value.trim();
            const submitBtn = contactForm.querySelector('button[type="submit"]');

            submitBtn.disabled = true;
            const originalBtnHTML = submitBtn.innerHTML;
            submitBtn.innerHTML = `
                <svg class="animate-spin -ml-1 mr-3 h-4.5 w-4.5 text-white inline-block" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span>Transmitting...</span>
            `;

            const formData = new FormData();
            formData.append('name', name);
            formData.append('email', email);
            formData.append('_replyto', email);
            formData.append('subject', subject);
            formData.append('message', message);
            formData.append('_subject', `Portfolio Contact: ${subject}`);
            formData.append('_autoresponse', `Hi ${name}, thanks for reaching out! I received your message and will get back to you shortly.`);

            try {
                const { isSuccess } = await submitPortfolioForm(formData);
                if (isSuccess) {
                    setContactFormStatus('Message delivered successfully. You should receive a confirmation email shortly.', false);
                    showToast('Message sent! It was delivered to Adarsh\'s inbox.');
                    contactForm.reset();
                } else {
                    setContactFormStatus('Delivery failed. Please email divaseadarsh608@gmail.com directly or try again in a moment.', true);
                    showToast('Could not deliver message. Try emailing directly.');
                }
            } catch (error) {
                setContactFormStatus('Network error. Please check your connection or email divaseadarsh608@gmail.com.', true);
                showToast('Network error while sending your message.');
                console.error(error);
            } finally {
                submitBtn.disabled = false;
                submitBtn.innerHTML = originalBtnHTML;
            }
        });
    }
});

// --- DYNAMIC MODAL SHOW/HIDE ---
const modal = document.getElementById('project-modal');
const modalContent = modal ? modal.querySelector('.modal-content') : null;

function showProjectModal(proj) {
    if (!modal || !modalContent) return;
    
    // Generate modal HTML structure
    modalContent.innerHTML = `
        <button id="close-modal" class="absolute top-4 right-4 text-slate-400 hover:text-white p-2 rounded-full hover:bg-slate-900 transition-colors z-30">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
        </button>
        <div class="space-y-8 relative z-10">
            <div class="space-y-2">
                <span class="text-xs font-mono text-cyan-400 font-bold bg-cyan-500/10 border border-cyan-500/20 px-2.5 py-1 rounded-full">Project Spec Sheet</span>
                <h2 class="text-2xl md:text-3xl font-black text-white mt-4">${proj.title}</h2>
                <p class="text-slate-400 font-mono text-xs md:text-sm text-cyan-500">${proj.subtitle}</p>
            </div>
            
            <div class="grid md:grid-cols-2 gap-8">
                <div class="space-y-6">
                    <div class="space-y-2">
                        <h4 class="text-xs font-mono text-slate-400 uppercase tracking-wider">The Problem</h4>
                        <p class="text-slate-300 text-sm leading-relaxed">${proj.details.problem}</p>
                    </div>
                    <div class="space-y-2">
                        <h4 class="text-xs font-mono text-slate-400 uppercase tracking-wider">The Solution</h4>
                        <p class="text-slate-300 text-sm leading-relaxed">${proj.details.solution}</p>
                    </div>
                    <div class="space-y-2">
                        <h4 class="text-xs font-mono text-slate-400 uppercase tracking-wider">Key Architectural Components</h4>
                        <ul class="space-y-2 text-slate-300 text-sm">
                            ${proj.details.architecture.map(step => `<li class="flex gap-2"><span class="text-cyan-400">✦</span> <span>${step}</span></li>`).join('')}
                        </ul>
                    </div>
                </div>
                
                <div class="space-y-6">
                    <div class="space-y-2">
                        <h4 class="text-xs font-mono text-slate-400 uppercase tracking-wider">Quantifiable Business Results</h4>
                        <ul class="space-y-2 text-slate-300 text-sm">
                            ${proj.details.results.map(res => `<li class="flex gap-2"><span class="text-emerald-400">✓</span> <span>${res}</span></li>`).join('')}
                        </ul>
                    </div>
                    <div class="space-y-2">
                        <h4 class="text-xs font-mono text-slate-400 uppercase tracking-wider">Representative Snippet</h4>
                        <pre class="p-4 rounded-xl bg-slate-950 border border-slate-900 overflow-x-auto text-[11px] font-mono text-cyan-300"><code class="language-python">${escapeHTML(proj.details.codeSnippet)}</code></pre>
                    </div>
                </div>
            </div>
            
            <div class="pt-6 border-t border-slate-900 flex gap-4">
                <a href="${proj.github}" target="_blank" class="bg-gradient-to-r from-cyan-500 to-indigo-500 hover:from-cyan-400 hover:to-indigo-400 text-white font-bold py-3 px-6 rounded-xl text-sm transition-all duration-300">Inspect Repo on GitHub</a>
            </div>
        </div>
    `;
    
    // Open modal with flex display
    modal.style.display = 'flex';
    document.body.style.overflow = 'hidden'; // Lock background scroll
    
    // Initialize highlighting for newly injected codeblock
    if (window.hljs) {
        window.hljs.highlightAll();
    }

    // Bind close click event
    const closeBtn = modalContent.querySelector('#close-modal');
    if (closeBtn) {
        closeBtn.addEventListener('click', hideProjectModal);
    }
}

function hideProjectModal() {
    if (!modal) return;
    modal.style.display = 'none';
    document.body.style.overflow = ''; // Unlock background scroll
}

// Click outside modal content to close it
window.addEventListener('click', (e) => {
    if (e.target === modal) {
        hideProjectModal();
    }
});

// ESC key to close modal
window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal.style.display === 'flex') {
        hideProjectModal();
    }
});

// Helper function to escape HTML special characters inside codeblocks
function escapeHTML(str) {
    return str.replace(/[&<>'"]/g, 
        tag => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            "'": '&#39;',
            '"': '&quot;'
        }[tag] || tag)
    );
}

// --- SYSTEM TELEMETRY TOAST NOTIFICATION UTILITY ---
function showToast(message) {
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'fixed bottom-6 right-6 z-50 space-y-3 pointer-events-none';
        document.body.appendChild(container);
    }
    
    const toast = document.createElement('div');
    toast.className = 'card-bg px-5 py-3.5 rounded-xl border border-cyan-500/20 text-xs font-mono text-slate-100 flex items-center gap-3 shadow-[0_4px_20px_rgba(6,182,212,0.15)] pointer-events-auto transition-all duration-300 transform translate-y-8 opacity-0';
    toast.innerHTML = `
        <span class="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping"></span>
        <span>[Telemetry] ${message}</span>
    `;
    container.appendChild(toast);
    
    // Animate in
    setTimeout(() => {
        toast.classList.remove('translate-y-8', 'opacity-0');
    }, 10);
    
    // Auto collapse after 4.5 seconds
    setTimeout(() => {
        toast.classList.add('translate-y-[-20px]', 'opacity-0');
        setTimeout(() => {
            toast.remove();
        }, 300);
    }, 4500);
}

// --- DYNAMIC HERO SUBTITLE ROLE ROTATOR ---
const roles = [
    "AI Full Stack Developer.",
    "AI Systems Engineer.",
    "Data Scientist & Analytics Architect.",
    "Python Backend Developer."
];

let roleIdx = 0;
let charIdx = 0;
let isDeleting = false;
const typingDelay = 100;
const erasingDelay = 50;
const newRoleDelay = 2000;

function typeRole() {
    const targetElement = document.getElementById('role-text');
    if (!targetElement) return;
    
    const currentRole = roles[roleIdx];
    
    if (isDeleting) {
        targetElement.innerText = currentRole.substring(0, charIdx - 1);
        charIdx--;
    } else {
        targetElement.innerText = currentRole.substring(0, charIdx + 1);
        charIdx++;
    }
    
    let delay = isDeleting ? erasingDelay : typingDelay;
    
    if (!isDeleting && charIdx === currentRole.length) {
        isDeleting = true;
        delay = newRoleDelay; // Pause before erasing
    } else if (isDeleting && charIdx === 0) {
        isDeleting = false;
        roleIdx = (roleIdx + 1) % roles.length;
        delay = 500; // Pause before typing next word
    }
    
    setTimeout(typeRole, delay);
}

// Trigger role rotator on load
setTimeout(typeRole, 1000);

// --- DUAL-COLOR CONSTALLATION PARTICLES CANVAS ANIMATION ---
const canvas = document.getElementById('bg-canvas');
if (canvas) {
    const ctx = canvas.getContext('2d');
    let particles = [];
    let width = canvas.width = window.innerWidth;
    let height = canvas.height = window.innerHeight;
    
    let mouse = { x: null, y: null, radius: 150 };
    
    window.addEventListener('mousemove', (e) => {
        mouse.x = e.clientX;
        mouse.y = e.clientY;
    });
    
    window.addEventListener('mouseout', () => {
        mouse.x = null;
        mouse.y = null;
    });
    
    window.addEventListener('resize', () => {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    });
    
    class Particle {
        constructor() {
            this.x = Math.random() * width;
            this.y = Math.random() * height;
            // Very slow, subtle drift speeds
            this.vx = (Math.random() - 0.5) * 0.35;
            this.vy = (Math.random() - 0.5) * 0.35;
            this.radius = Math.random() * 1.5 + 0.5;
            // Randomly designate cyan or purple particles
            this.color = Math.random() > 0.5 ? 'rgba(6, 182, 212, 0.45)' : 'rgba(168, 85, 247, 0.45)';
        }
        
        update() {
            this.x += this.vx;
            this.y += this.vy;
            
            // Boundary bounce checks
            if (this.x < 0 || this.x > width) this.vx = -this.vx;
            if (this.y < 0 || this.y > height) this.vy = -this.vy;
            
            // Mouse push deflection effect
            if (mouse.x !== null && mouse.y !== null) {
                const dx = this.x - mouse.x;
                const dy = this.y - mouse.y;
                const dist = Math.sqrt(dx*dx + dy*dy);
                if (dist < mouse.radius) {
                    const force = (mouse.radius - dist) / mouse.radius;
                    // Deflect particles slightly away from cursor coordinates
                    this.x += (dx / dist) * force * 1.2;
                    this.y += (dy / dist) * force * 1.2;
                }
            }
        }
        
        draw() {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            ctx.fillStyle = this.color;
            ctx.fill();
        }
    }
    
    function init() {
        particles = [];
        // Match density with screen width dimensions
        const density = Math.floor((width * height) / 11000);
        for (let i = 0; i < Math.min(density, 120); i++) {
            particles.push(new Particle());
        }
    }
    
    function animate() {
        ctx.clearRect(0, 0, width, height);
        
        particles.forEach(p => {
            p.update();
            p.draw();
        });
        
        // Draw constellation link lines between close nodes
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx*dx + dy*dy);
                
                if (dist < 110) {
                    // Compute link opacity based on distance
                    const alpha = (110 - dist) / 110 * 0.12;
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    // Line color blends with endpoints
                    ctx.strokeStyle = `rgba(99, 102, 241, ${alpha})`;
                    ctx.lineWidth = 0.75;
                    ctx.stroke();
                }
            }
        }
        
        requestAnimationFrame(animate);
    }
    
    init();
    animate();
}

// --- CUSTOM CYBER CURSOR INTEGRATION ---
(function() {
    const cursor = document.getElementById('cyber-cursor');
    const follower = document.getElementById('cyber-cursor-follower');
    if (!cursor || !follower) return;

    let posX = 0, posY = 0;
    let mouseX = 0, mouseY = 0;

    document.addEventListener('mousemove', (e) => {
        mouseX = e.clientX;
        mouseY = e.clientY;
        cursor.style.transform = `translate3d(${mouseX}px, ${mouseY}px, 0)`;
    });

    function animateCursor() {
        posX += (mouseX - posX) * 0.15;
        posY += (mouseY - posY) * 0.15;
        follower.style.transform = `translate3d(${posX}px, ${posY}px, 0)`;
        requestAnimationFrame(animateCursor);
    }
    animateCursor();

    function updateInteractives() {
        const interactives = document.querySelectorAll('a, button, input, textarea, .project-card, .work-project-card, .poll-option');
        interactives.forEach(el => {
            if (el.dataset.cursorBound) return;
            el.dataset.cursorBound = "true";
            
            el.addEventListener('mouseenter', () => {
                document.body.classList.add('cursor-hover');
            });
            el.addEventListener('mouseleave', () => {
                document.body.classList.remove('cursor-hover');
            });
        });
    }

    updateInteractives();
    setInterval(updateInteractives, 1000);
})();
