const RESUME_PDF = 'ADresume.pdf';
const CONTACT_EMAIL = 'divaseadarsh608@gmail.com';

// 1. Open https://web3forms.com  2. Enter divaseadarsh608@gmail.com  3. Paste the access key from your email below:
const WEB3FORMS_ACCESS_KEY = 'fa7c57c1-8653-4553-aebc-7cc02bebe1ab';
const WEB3FORMS_ENDPOINT = 'https://api.web3forms.com/submit';

const experiences = [
    {
        period: 'Sep 2025 - Present',
        role: 'AI Full Stack Developer',
        company: 'Allwave AV Systems Pvt Ltd',
        location: 'Mumbai, Maharashtra',
        highlights: [
            'Lead development of <strong>six production platforms</strong> (inventory &amp; compliance, AVIXA presales AI, support RAG, programmer assistant, HR automation, 3D configurator) used across Allwave operations.',
            'Full-stack delivery: <strong>TypeScript/React</strong>, <strong>Python/FastAPI</strong>, AWS Amplify Gen 2, App Runner, Lambda, DynamoDB, Cognito, SES, GraphQL/AppSync, Gemini, and LangChain RAG.',
            'End-to-end ownership—CDK/CI/CD, multi-tenant RBAC, production observability, and India compliance (GST, MSME, Tally XML). Details in <a href="#production-systems" class="text-indigo-400 hover:text-indigo-300 font-semibold">Production Systems</a>.'
        ]
    },
    {
        period: 'Dec 2024 - Present',
        role: 'Python Backend Developer Intern',
        company: 'Aceminds Digital Pvt Ltd',
        location: 'Pune, Maharashtra',
        highlights: [
            'Built and maintained production <strong>FastAPI</strong> and <strong>Flask</strong> services with ML/DL inference integrated into live API workflows.',
            'Partnered with frontend engineers to ship responsive UIs connected to model backends—foundation for my current full-stack role at Allwave.',
            'Designed database schemas and optimized queries, achieving a <strong>15% reduction in data retrieval times</strong>.'
        ]
    }
];

// Rank 1 = highest impact / architectural scale (shown first to hiring managers)
const workProjects = [
    {
        id: 'av-inventory',
        rank: 1,
        title: 'AV Inventory Ops',
        subtitle: 'Enterprise Operations Platform — Inventory, Billing, GST/MSME Compliance & Tally',
        company: 'Allwave AV Systems',
        period: 'Sep 2025 – Present',
        summary: 'Flagship back-office system replacing spreadsheets with a governed AWS platform: 36 DynamoDB tables, 19 Lambda functions, GraphQL API, India tax compliance, and a React PWA used daily by operations and finance.',
        description: 'End-to-end AV integration operations platform (ap-south-1) built on AWS Amplify Gen 2. Unifies inventory, procurement, invoicing, AMC/service tickets, client portal, and statutory compliance (GST, MSME, Tally XML) in one auditable system with scheduled automation and AI assistance.',
        tags: ['TypeScript', 'Amplify Gen 2', 'GraphQL', 'DynamoDB', 'Lambda', 'CDK', 'SES', 'OpenSearch', 'Playwright'],
        stack: 'TypeScript 5.7 · Amplify Gen 2 · aws-cdk-lib · AppSync GraphQL · DynamoDB (36 tables) · 19× Lambda · Cognito · S3 · SES/SESv2 · SNS · EventBridge · Scheduler · Secrets Manager · OpenSearch · React 18 PWA · TanStack Query · Zustand · Playwright E2E',
        architecture: [
            'Domain model spans ProductMaster, UnitRecord (8 GSIs), GRN, Delivery Challan, invoices, purchase orders, BOQ uploads, AMC, service tickets, vendors, clients, projects, HSN master, audit logs, chat sessions, and financial-year counters.',
            '19 purpose-built Lambdas: alert-engine, reminder-dispatcher, invoice-scheduler, payment-reminder-sender, msme-compliance-checker, boq-parser, chatbot-handler, tally-export-generator, forex-rate-fetcher, depreciation-engine, hsn-validator, fy-rollover, tds-auto-creator, warranty-alert-monthly, amc-renewal-checker, daily-digest, client-portal-handler, user-admin.',
            'Shared TypeScript utilities for FY boundaries, GSTIN validation, HSN codes, invoice numbering, and Tally XML generation—single source of truth for finance handoff.',
            'React PWA: shadcn/Radix UI, react-hook-form + Zod, TanStack Table/Virtual, Recharts, html5-qrcode scanning, Cmd+K command palette, session idle monitor, floating Gemini assistant on core screens.'
        ],
        infrastructure: [
            'Production hardening: PITR, encryption at rest, deletion protection on all tables; WAF on public endpoints.',
            'Secrets Manager for Gemini and ExchangeRate-API keys; SES production with DKIM, SPF, and DMARC.',
            'EventBridge/Scheduler: FY rollover (1 Apr IST), monthly TDS, warranty alerts, AMC renewal, daily digest emails.',
            '22 transactional SES HTML templates (build-ses-templates.ts); Vitest on backend, Playwright mock/live E2E on frontend.'
        ],
        apis: [
            'AppSync GraphQL CRUD across inventory, billing, procurement, compliance, and client-portal modules.',
            'On-demand Lambda invocations documented in av-inventory.postman_collection.json.',
            'Tally XML export and client-portal token handlers for finance and customer self-service.'
        ],
        highlights: [
            'Designed and shipped a 36-table data model with GSIs tuned for high-volume unit tracking and reporting.',
            'Automated India compliance workflows (MSMED Act 2006, GSTIN validation, e-Way Bill alignment, Udyam certs on S3).',
            'BOQ parser Lambda plus embedded Gemini chatbot reduce manual data entry across operations screens.',
            'Delivered finance-ready exports and scheduled jobs that eliminate recurring manual GST/MSME risk.'
        ],
        impact: [
            'Single system of record for Allwave back-office—replacing fragmented spreadsheets with full audit trails.',
            'Demonstrates ability to own large-scale serverless architecture, compliance, and production operations—not just features.'
        ]
    },
    {
        id: 'bingo',
        rank: 2,
        title: 'BINGO',
        subtitle: 'AVIXA-Compliant AI Bill of Quantities Platform (v2.1)',
        company: 'Allwave AV Systems',
        period: 'Sep 2025 – Present',
        summary: 'Presales flagship: AI-generated, AVIXA-validated BOQs from 2,000+ curated products—with RAG assistant, brand enforcement, CAD/DXF export, and async Gemini jobs on Amplify Gen 2.',
        description: 'Bill of Quantities Intelligent Next-Gen Optimizer—enterprise presales software that turns client requirements into AVIXA CTS-D compliant BOQs, schematics, and client-ready proposals. Combines structured questionnaires, grounded RAG, engineering calculators, and export pipelines used by global integrator presales teams.',
        tags: ['TypeScript', 'React', 'Amplify Gen 2', 'CDK', 'DynamoDB', 'Cognito', 'Gemini', 'RAG', 'SES', 'DXF'],
        stack: 'TypeScript 5 · React 18 · Vite · AWS Amplify Gen 2 · aws-cdk-lib · DynamoDB · AppSync GraphQL · Cognito · Lambda · SES · Google Gemini 2.5 Pro · Zod · ExcelJS · jsPDF · docx',
        architecture: [
            'Amplify Gen 2 backend: auth, GraphQL models, 8+ Lambdas (gemini-proxy, job starter/worker/status, weekly-report, send-proposal, bootstrap-admin, user-profile, sync-user-access).',
            'Ask BINGO RAG: vector + legacy DB search, IndexedDB cache, exact-model matching, and controlled web fallback—grounded answers, not hallucinated SKUs.',
            'geminiService.ts: AV consultant prompts, BOQ refinement, product DB grounding, and category/size-aware query routing.',
            'AVIXA engines: DMD viewing distance, ACE audio coverage, VIP camera FOV; per-category brand enforcement.',
            'CAD pipeline: signal flow, rack elevations, Stardraw symbols, DXF export with 27+ layers and cable/equipment schedules.'
        ],
        infrastructure: [
            'Async Gemini job queue to avoid API Gateway timeouts on large BOQ generations.',
            'EventBridge weekly SES activity reports to operations distribution lists.',
            'Cognito groups + bootstrap-admin Lambda—no API keys exposed on the client.',
            'IAM-scoped SES; DynamoDB ActivityLog for enterprise audit analytics.'
        ],
        apis: [
            'AppSync GraphQL for products, BOQs, users, activity logs, and proposals.',
            'Lambda-backed Gemini proxy and long-running job status polling.',
            'Client services: productService, activityLogService, userManagementService, askBingoService.'
        ],
        highlights: [
            'Database-first product sourcing with web price estimation and automatic dimension retrieval.',
            'Natural-language BOQ edits (e.g. “add 2 more speakers”) with strict brand rules per component class.',
            'Multi-format export: ExcelJS, jsPDF, and docx for client-ready proposal packs.',
            'Secure deploy automation (deploy:secure) and Cognito group sync (sync-cognito-groups.mjs).'
        ],
        impact: [
            'Collapses multi-day manual BOQ work into AVIXA-validated, export-ready packages—direct revenue impact for presales.',
            'Showcases full-stack ownership: AI/RAG, domain engineering rules, CDK infra, and RBAC at enterprise scale.'
        ]
    },
    {
        id: 'pronto',
        rank: 3,
        title: 'PRONTO',
        subtitle: 'Enterprise Support Brain — RAG, Jira Automation & Field Voice',
        company: 'Allwave AV Systems',
        period: 'Sep 2025 – Present',
        summary: 'Production support platform: FAISS RAG over internal KB, LangChain + Gemini, Jira ticket intelligence, bilingual voice I/O, and one-click AI solutions posted back to tickets.',
        description: 'PRONTO is Allwave’s production support copilot—used by L1/L2 engineers to resolve AV integration issues faster. Grounds every answer in internal documentation, integrates with Jira for ticket context, and automates analysis comments while keeping humans in the loop.',
        tags: ['Python', 'FastAPI', 'TypeScript', 'React', 'FAISS', 'LangChain', 'Gemini', 'Jira', 'Lambda', 'boto3'],
        stack: 'Python 3.10+ · FastAPI · Uvicorn · LangChain · langchain-google-genai · FAISS · React 18 · Vite · Gemini · Jira REST · boto3 · PyJWT · bcrypt · AWS Lambda · Amplify',
        architecture: [
            'Lazy-loaded RAGEngine, JiraService, and EscalationService to stay within Lambda cold-start INIT limits.',
            'FAISS vector index (local + S3 sync) with ingestion from curated data/ knowledge base.',
            'SupportWorkflow state machine: smart chat, ticket analysis, apply-solution, and feedback loops.',
            'ADF (Atlassian Document Format) parser for rich Jira comments and AI-generated summaries.',
            'React admin UI: ticket pagination, analytics, browser notifications for high-priority incidents.'
        ],
        infrastructure: [
            'AWS Lambda + API Gateway; FAISS hydration from S3 on cold start.',
            'Amplify frontend → API Gateway prod stage with strict production CORS and auth.',
            'JWT auth with admin approval gate, bcrypt hashing, and secure session patterns.',
            'BackgroundTasks for long Jira posts and streaming where applicable.'
        ],
        apis: [
            'Smart chat, image chat, ticket chat, escalation, feedback, and analytics endpoints.',
            'Jira: paginated fetch, AI comments prefixed [PRONTO - AI Analysis], customer send-back flows.',
            'KB rebuild, health checks, webhooks, schematic generation, Hindi/English voice pipelines.'
        ],
        highlights: [
            'Shipped and operate in production—real users, real incidents, and production SLAs.',
            'LangChain + Gemini chains with AV-domain system instructions and citation-style grounding.',
            'Resolved production issues: Lambda INIT timeouts, S3 KB sync, cross-environment networking.',
            'PRONTO-branded email templates for onboarding and password reset.'
        ],
        impact: [
            'Measurable reduction in mean-time-to-resolution via RAG-cited answers and automated ticket commentary.',
            'Proof of shipping AI to production with security, observability, and third-party integrations (Jira).'
        ]
    },
    {
        id: 'nexo',
        rank: 4,
        title: 'Nexo',
        subtitle: 'AV Programmer Assistant — BOQ-to-Code & Control System Guides',
        company: 'Allwave AV Systems',
        period: 'Sep 2025 – Present',
        summary: 'Production TypeScript/React platform for Crestron, Extron, QSC, and allied stacks—BOQ import, Gemini code generation, troubleshooting modes, and team snippet libraries on AWS App Runner.',
        description: 'Nexo accelerates AV programmers from BOQ to deployable control code. Combines structured capture (site, network, audio/video/lighting handover) with free-text AI modes, persisted history, and admin-provisioned access—reducing rework across distributed programming teams.',
        tags: ['TypeScript', 'React', 'Vite', 'Node.js', 'App Runner', 'Cognito', 'DynamoDB', 'S3', 'Gemini', 'XLSX'],
        stack: 'TypeScript · React 18 · Vite · Tailwind · Node.js HTTP API · @google/genai · AWS App Runner · Cognito · DynamoDB · S3 · mammoth · xlsx',
        architecture: [
            'Dual UX: Quick Ask (generate / troubleshoot / document) and BOQ Guided Flow (parse → questions → guide).',
            'Pipeline: Excel/Word/text → /api/boq/parse → validation → /api/boq/questions → /api/boq/guide.',
            'Structured prompts for platform, task type, and site/network/av handover fields.',
            'Per-user history and team snippets in DynamoDB (PK userId, SK timestamp); admin provisioning UI.'
        ],
        infrastructure: [
            'AWS App Runner + ECR; secrets via environment (GEMINI_API_KEY); /api/health for monitoring.',
            'Cognito invite-only pools with ADMIN RBAC restored via CLI and admin UI after incidents.',
            'Lambda-ready modules (ask, history, snippets) with API Gateway authorizer pattern.',
            'Security response: key rotation, CloudTrail analysis, verified clean Git history.'
        ],
        apis: [
            'POST /api/ask — mode-specific Gemini prompts (generate, troubleshoot, document).',
            'POST /api/boq/parse, /api/boq/questions, /api/boq/guide — multi-step BOQ programming.',
            'GET/DELETE /api/history · GET/POST /api/snippets — session persistence and team knowledge.'
        ],
        highlights: [
            'Deployed to production with syntax-highlighted, deployment-ready code output.',
            'BOQ parsers for .xlsx/.xls/.docx with intelligent column detection.',
            'Diagnosed 502/timeouts: App Runner 120s limit vs long Gemini jobs—CloudTrail + Envoy headers.',
            'Zero client-side API keys in production builds.'
        ],
        impact: [
            'Cuts BOQ-to-code cycle time for programming teams across brands (Crestron, Extron, QSC, etc.).',
            'Demonstrates product thinking for specialist users—not generic chat, but workflow-shaped AI.'
        ]
    },
    {
        id: 'hiro',
        rank: 5,
        title: 'HiRo',
        subtitle: 'AI HR Platform — Recruitment, Onboarding & Email Automation',
        company: 'Allwave AV Systems',
        period: 'Sep 2025 – Present',
        summary: 'Multi-tenant HR platform with recruitment pipelines, onboarding workflows, Gemini HR copilot, and corporate SES email—containerized on App Runner with CodeBuild → ECR CI/CD.',
        description: 'HiRo centralizes hiring and onboarding for Allwave: branded templates, workflow vs campaign automation, per-user verified sending domains, and auditable activity—built as a secure multi-tenant SaaS on AWS.',
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
            'Unified hiring and onboarding with enterprise email deliverability and repeatable releases.',
            'Demonstrates internal SaaS delivery: multi-tenant data, CI/CD, and production incident response.'
        ]
    },
    {
        id: 'ez-configurator',
        rank: 6,
        title: 'EZ Configurator',
        subtitle: '3D Room Design Studio — AVIXA Physics, Gemini Auto-Design & PDF Handover',
        company: 'Allwave AV Systems',
        period: 'Sep 2025 – Present',
        summary: 'Browser-based design studio: React-Three-Fiber 3D, DISCAS/SPL/RT60 physics engines, 42-SKU catalog, Gemini auto-design API, and one-click PDF handover (compliance + GST BOQ + network scheme).',
        description: 'EZ Configurator lets presales and engineering validate room designs before procurement. Combines real AVIXA calculations, immersive 3D placement, and AI-assisted layout proposals—outputting client-ready documentation from a single workflow.',
        tags: ['JavaScript', 'React', 'Three.js', 'R3F', 'Vite', 'Express', 'Gemini', 'AVIXA', 'Zustand'],
        stack: 'React 18 · Vite · Three.js · @react-three/fiber · @react-three/drei · postprocessing · Zustand · Tailwind · Express · @google/generative-ai · jsPDF · html2canvas · Framer Motion',
        architecture: [
            'engines/physics.js: DISCAS viewing distance, SPL (Lp = Lw − 20log₁₀(r) − 11), Sabine RT60, PAG/NAG feedback, cable limits (HDMI, HDBaseT, USB, Dante, AVoIP).',
            'Global Zustand store; data/products.js with 42 SKUs across 14 brands; catalog build/validate scripts.',
            '3D: PBR wall materials (6 absorption coefficients), snap-to-surface placement, SSAO/Bloom/Vignette post-processing.',
            'Express sidecar: secure Gemini keys, POST /api/chat and POST /api/auto-design for constraint-driven room proposals.'
        ],
        infrastructure: [
            'Split Vite frontend + Node Express AI API (port 8787) for secure key isolation in UAT/production patterns.',
            'Structured Gemini error codes surfaced in UI for supportability.',
            'PDF pipeline: AVIXA compliance report, GST BOQ, and IP addressing scheme in one export pack.'
        ],
        apis: [
            'GET /health — AI sidecar health check.',
            'POST /api/chat — conversational AV design assistant.',
            'POST /api/auto-design — automated layout and equipment proposals from constraints.'
        ],
        highlights: [
            'Floor legibility heatmaps and per-display AVIXA pass/fail badges—evidence before purchase orders.',
            'Framer Motion UX; remark-gfm markdown for AI-generated design narratives.',
            'catalog:build and catalog:validate scripts enforce product taxonomy integrity.',
            'Module 8 handover unifies compliance, BOQ, and network documentation for sign-off.'
        ],
        impact: [
            'De-risks hardware commits by proving designs against physics—not slides alone.',
            'Highlights rare blend of 3D graphics, domain engineering math, and generative AI in one product.'
        ]
    }
];

function getSortedWorkProjects() {
    return workProjects.slice().sort((a, b) => a.rank - b.rank);
}

async function submitPortfolioForm({ name, email, subject, message, fromName }) {
    if (!WEB3FORMS_ACCESS_KEY) {
        return {
            isSuccess: false,
            payload: {
                message: 'Add your Web3Forms access key in portfolio-extensions.js (free at web3forms.com).'
            },
            status: 0
        };
    }

    const response = await fetch(WEB3FORMS_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({
            access_key: WEB3FORMS_ACCESS_KEY,
            name,
            email,
            subject,
            message,
            from_name: fromName || 'Adarsh Portfolio',
            botcheck: ''
        })
    });

    let payload = {};
    try {
        payload = await response.json();
    } catch {
        payload = { message: 'Invalid response from mail service.' };
    }

    const isSuccess = response.ok && payload.success === true;
    return { isSuccess, payload, status: response.status };
}

function setFormStatus(elementId, message, isError = false) {
    const statusEl = document.getElementById(elementId);
    if (!statusEl) return;
    statusEl.textContent = message;
    statusEl.classList.remove('hidden', 'contact-status-success', 'contact-status-error');
    statusEl.classList.add(isError ? 'contact-status-error' : 'contact-status-success');
}

function setContactFormStatus(message, isError = false) {
    setFormStatus('contact-form-status', message, isError);
}

function setAmaFormStatus(message, isError = false) {
    setFormStatus('ama-form-status', message, isError);
}

function showToast(message) {
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'fixed bottom-6 right-6 z-50 space-y-3 pointer-events-none';
        document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    toast.className = 'card-bg px-5 py-3 rounded-xl border border-indigo-500/30 text-sm text-slate-100 shadow-lg pointer-events-auto transition-all duration-300 transform translate-y-4 opacity-0';
    toast.textContent = message;
    container.appendChild(toast);

    requestAnimationFrame(() => toast.classList.remove('translate-y-4', 'opacity-0'));
    setTimeout(() => {
        toast.classList.add('opacity-0');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function renderExperienceTimeline() {
    const timeline = document.getElementById('experience-timeline');
    if (!timeline) return;

    timeline.innerHTML = '<div class="absolute left-1/2 -translate-x-1/2 h-full w-0.5 bg-slate-700"></div>';

    experiences.forEach((exp, index) => {
        const isLeft = index % 2 === 0;
        const item = document.createElement('div');
        item.className = 'relative mb-12';
        item.innerHTML = `
            <div class="absolute left-1/2 -translate-x-1/2 w-4 h-4 bg-indigo-500 rounded-full mt-1.5 ring-8 ring-gray-900 z-10"></div>
            <div class="${isLeft ? 'ml-auto md:ml-0 md:w-[45%] md:mr-[5%]' : 'md:ml-[55%] md:w-[45%]'} card-bg p-6 rounded-2xl shadow-lg">
                <time class="text-sm font-semibold text-indigo-400">${exp.period}</time>
                <h3 class="text-xl font-bold text-white mt-1">${exp.role}</h3>
                <p class="text-slate-400 mb-4">${exp.company} (${exp.location})</p>
                <ul class="list-none space-y-2 text-slate-300 text-sm">
                    ${exp.highlights.map(point => `<li class="flex items-start"><span class="text-indigo-400 mr-2 shrink-0">✓</span><span>${point}</span></li>`).join('')}
                </ul>
            </div>
        `;
        timeline.appendChild(item);
    });
}

function renderProductionSystems() {
    const grid = document.getElementById('production-systems-grid');
    if (!grid) return;

    grid.innerHTML = '';
    getSortedWorkProjects().forEach((proj) => {
        const card = document.createElement('article');
        card.className = 'production-card project-card card-bg rounded-2xl flex flex-col overflow-hidden transform hover:-translate-y-2 transition-transform duration-300 cursor-pointer';
        const summary = proj.summary || proj.description || '';
        card.innerHTML = `
            <div class="p-6 flex flex-col flex-grow">
                <div class="flex items-center gap-3 mb-2">
                    <span class="production-card__rank" aria-hidden="true">${String(proj.rank).padStart(2, '0')}</span>
                    <h3 class="text-xl font-bold text-white">${proj.title}</h3>
                </div>
                <p class="text-indigo-300 text-sm mb-2 leading-snug">${proj.subtitle}</p>
                <p class="text-slate-400 mb-4 text-sm leading-relaxed flex-grow">${summary}</p>
                <div class="flex flex-wrap gap-2 mt-auto pt-2">
                    ${proj.tags.slice(0, 4).map(tag => `<span class="tag rounded-md px-2 py-1 text-xs">${tag}</span>`).join('')}
                </div>
            </div>`;
        card.setAttribute('role', 'button');
        card.setAttribute('tabindex', '0');
        card.setAttribute('aria-label', `${proj.title}, ranked ${proj.rank}. Click for full technical details.`);
        card.addEventListener('click', () => showWorkProjectModal(proj));
        card.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                showWorkProjectModal(proj);
            }
        });
        grid.appendChild(card);
    });
}

function showWorkProjectModal(proj) {
    const modal = document.getElementById('project-modal');
    const modalContent = modal?.querySelector('.modal-content');
    if (!modal || !modalContent) return;

    const rankLabel = proj.rank ? `<span class="text-xs font-bold text-indigo-400 bg-indigo-500/10 border border-indigo-500/25 px-2 py-0.5 rounded-full mr-2">Rank #${proj.rank}</span>` : '';

    const listSection = (title, items, bulletClass) => {
        if (!items || !items.length) return '';
        return `
            <div class="mb-6">
                <h4 class="text-lg font-semibold text-white mb-2">${title}</h4>
                <ul class="list-none space-y-2">
                    ${items.map(item => `<li class="flex items-start text-slate-300"><span class="${bulletClass} mr-3 mt-1 shrink-0">▪</span><span>${item}</span></li>`).join('')}
                </ul>
            </div>`;
    };

    modalContent.innerHTML = `
        <button class="absolute top-4 right-6 text-slate-400 hover:text-white text-3xl z-10" type="button" onclick="closeModal()">&times;</button>
        <div class="px-1">
            <div class="flex flex-wrap items-center gap-2 mb-2">${rankLabel}<span class="text-sm text-indigo-400 font-semibold">${proj.company} · ${proj.period}</span></div>
            <h2 class="text-3xl font-bold text-white mb-1">${proj.title}</h2>
            <p class="text-indigo-300 mb-3">${proj.subtitle}</p>
            ${proj.summary ? `<p class="production-modal__summary">${proj.summary}</p>` : ''}
            <p class="text-slate-300 mb-6 leading-relaxed">${proj.description}</p>
            <div class="mb-6 card-bg p-4 rounded-xl border border-slate-700">
                <h4 class="text-lg font-semibold text-white mb-2">Full Tech Stack</h4>
                <p class="text-slate-300 text-sm">${proj.stack}</p>
            </div>
            ${listSection('System Architecture', proj.architecture, 'text-indigo-400')}
            ${listSection('AWS & Infrastructure', proj.infrastructure, 'text-indigo-400')}
            ${listSection('APIs & Integrations', proj.apis, 'text-indigo-400')}
            ${listSection('Engineering Highlights', proj.highlights, 'text-indigo-400')}
            ${listSection('Business Impact', proj.impact, 'text-emerald-400')}
            <h4 class="text-lg font-semibold text-white mb-3">Technologies</h4>
            <div class="flex flex-wrap gap-2">${proj.tags.map(tag => `<span class="tag rounded-md px-3 py-1 text-sm">${tag}</span>`).join('')}</div>
        </div>`;

    modal.style.display = 'flex';
    document.body.style.overflow = 'hidden';
}

const PDFJS_CDN = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174';

const ResumeViewer = {
    pdfDoc: null,
    zoom: 1,
    baseScale: 1,
    renderToken: 0,
    pdfJsPromise: null,

    loadPdfJs() {
        if (window.pdfjsLib) return Promise.resolve(window.pdfjsLib);
        if (this.pdfJsPromise) return this.pdfJsPromise;

        this.pdfJsPromise = new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = `${PDFJS_CDN}/pdf.min.js`;
            script.onload = () => {
                pdfjsLib.GlobalWorkerOptions.workerSrc = `${PDFJS_CDN}/pdf.worker.min.js`;
                resolve(pdfjsLib);
            };
            script.onerror = () => reject(new Error('Failed to load PDF viewer'));
            document.head.appendChild(script);
        });

        return this.pdfJsPromise;
    },

    setLoading(visible) {
        const loading = document.getElementById('resume-loading');
        if (loading) loading.hidden = !visible;
    },

    updateZoomLabel() {
        const label = document.getElementById('resume-zoom-label');
        if (label) label.textContent = `${Math.round(this.zoom * 100)}%`;
    },

    async open() {
        const resumeModal = document.getElementById('resume-modal');
        const downloadBtn = document.getElementById('resume-download-btn');
        const pagesEl = document.getElementById('resume-pages');
        if (!resumeModal || !pagesEl) return;

        if (downloadBtn) downloadBtn.href = RESUME_PDF;
        resumeModal.style.display = 'flex';
        resumeModal.setAttribute('aria-hidden', 'false');
        document.body.style.overflow = 'hidden';

        this.setLoading(true);
        pagesEl.innerHTML = '';
        const token = ++this.renderToken;

        try {
            await this.loadPdfJs();
            if (!this.pdfDoc) {
                const loadingTask = pdfjsLib.getDocument(RESUME_PDF);
                this.pdfDoc = await loadingTask.promise;
            }
            if (token !== this.renderToken) return;

            this.zoom = 1;
            await this.renderPages();
            this.updateZoomLabel();
        } catch (err) {
            console.error(err);
            pagesEl.innerHTML = '<p class="resume-viewer__error">Could not load resume. Use Download PDF instead.</p>';
        } finally {
            if (token === this.renderToken) this.setLoading(false);
        }
    },

    close() {
        const resumeModal = document.getElementById('resume-modal');
        if (!resumeModal) return;
        resumeModal.style.display = 'none';
        resumeModal.setAttribute('aria-hidden', 'true');
        document.body.style.overflow = '';
        this.renderToken++;
        const pagesEl = document.getElementById('resume-pages');
        if (pagesEl) pagesEl.innerHTML = '';
        this.setLoading(false);
    },

    async renderPages() {
        const pagesEl = document.getElementById('resume-pages');
        const stage = document.getElementById('resume-viewer-stage');
        if (!pagesEl || !this.pdfDoc || !stage) return;

        const token = this.renderToken;
        pagesEl.innerHTML = '';

        const stageWidth = Math.max(stage.clientWidth - 32, 280);
        const firstPage = await this.pdfDoc.getPage(1);
        const defaultViewport = firstPage.getViewport({ scale: 1 });
        this.baseScale = stageWidth / defaultViewport.width;

        for (let num = 1; num <= this.pdfDoc.numPages; num++) {
            if (token !== this.renderToken) return;

            const page = await this.pdfDoc.getPage(num);
            const scale = this.baseScale * this.zoom * window.devicePixelRatio;
            const viewport = page.getViewport({ scale });

            const sheet = document.createElement('div');
            sheet.className = 'resume-page';
            sheet.setAttribute('data-page', String(num));

            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = viewport.width;
            canvas.height = viewport.height;
            canvas.style.width = `${viewport.width / window.devicePixelRatio}px`;
            canvas.style.height = `${viewport.height / window.devicePixelRatio}px`;

            sheet.appendChild(canvas);
            pagesEl.appendChild(sheet);

            await page.render({ canvasContext: ctx, viewport }).promise;
        }
    },

    async setZoom(nextZoom) {
        this.zoom = Math.min(1.75, Math.max(0.65, nextZoom));
        this.updateZoomLabel();
        this.setLoading(true);
        const token = ++this.renderToken;
        try {
            await this.renderPages();
        } finally {
            if (token === this.renderToken) this.setLoading(false);
        }
    },

    zoomIn() {
        return this.setZoom(this.zoom + 0.15);
    },

    zoomOut() {
        return this.setZoom(this.zoom - 0.15);
    },

    fitWidth() {
        return this.setZoom(1);
    }
};

function showResumeModal() {
    ResumeViewer.open();
}

function hideResumeModal() {
    ResumeViewer.close();
}

function initResumeViewer() {
    const zoomIn = document.getElementById('resume-zoom-in');
    const zoomOut = document.getElementById('resume-zoom-out');
    const fitWidth = document.getElementById('resume-fit-width');

    if (zoomIn) zoomIn.addEventListener('click', () => ResumeViewer.zoomIn());
    if (zoomOut) zoomOut.addEventListener('click', () => ResumeViewer.zoomOut());
    if (fitWidth) fitWidth.addEventListener('click', () => ResumeViewer.fitWidth());

    let resizeTimer;
    window.addEventListener('resize', () => {
        const resumeModal = document.getElementById('resume-modal');
        if (!resumeModal || resumeModal.style.display !== 'flex' || !ResumeViewer.pdfDoc) return;
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => {
            ResumeViewer.fitWidth();
        }, 200);
    });
}

function initForms() {
    const commentForm = document.getElementById('comment-form');
    if (commentForm) {
        commentForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const honey = document.getElementById('ama-honey');
            if (honey && honey.value.trim()) return;

            const name = document.getElementById('comment-name').value.trim();
            const email = document.getElementById('comment-email').value.trim();
            const text = document.getElementById('comment-text').value.trim();
            const submitBtn = commentForm.querySelector('button[type="submit"]');
            submitBtn.disabled = true;
            const originalHTML = submitBtn.innerHTML;
            submitBtn.textContent = 'Sending...';

            submitPortfolioForm({
                name,
                email,
                subject: `Portfolio AMA Question from ${name}`,
                message: text,
                fromName: 'Portfolio — Ask Me Anything'
            })
                .then(({ isSuccess, payload }) => {
                    if (isSuccess) {
                        setAmaFormStatus('Question sent! It was delivered to Adarsh\'s inbox.', false);
                        showToast('Question sent successfully!');
                        commentForm.reset();
                    } else {
                        const hint = payload?.message ? ` ${payload.message}` : '';
                        setAmaFormStatus(`Could not send.${hint} Email divaseadarsh608@gmail.com directly.`, true);
                    }
                })
                .catch(() => setAmaFormStatus('Network error. Please try again or email divaseadarsh608@gmail.com.', true))
                .finally(() => {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = originalHTML;
                });
        });
    }

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
            const originalHTML = submitBtn.innerHTML;
            submitBtn.textContent = 'Sending...';

            try {
                const { isSuccess, payload } = await submitPortfolioForm({
                    name,
                    email,
                    subject: `Portfolio Contact: ${subject}`,
                    message: `Subject: ${subject}\n\n${message}`,
                    fromName: 'Portfolio — Get In Touch'
                });
                if (isSuccess) {
                    setContactFormStatus('Message delivered to divaseadarsh608@gmail.com. Thank you!', false);
                    showToast('Message sent!');
                    contactForm.reset();
                } else {
                    const hint = payload?.message ? ` ${payload.message}` : '';
                    setContactFormStatus(`Delivery failed.${hint} Please email divaseadarsh608@gmail.com directly.`, true);
                }
            } catch {
                setContactFormStatus('Network error. Please check your connection or email directly.', true);
            } finally {
                submitBtn.disabled = false;
                submitBtn.innerHTML = originalHTML;
            }
        });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    if (!window.location.hash) {
        window.scrollTo(0, 0);
    }

    renderExperienceTimeline();
    renderProductionSystems();
    renderTechnicalNotes();
    renderNotesQuickLinks();
    initForms();
    initResumeViewer();

    const viewResumeBtn = document.getElementById('view-resume-btn');
    const closeResumeBtn = document.getElementById('close-resume-modal');
    const resumeModal = document.getElementById('resume-modal');

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
});
