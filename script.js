<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Adarsh Divase - AI & Data Science Portfolio</title>
    
    <!-- Favicon -->
    <link rel="icon" type="image/x-icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🤖</text></svg>">
    
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        primary: '#6366f1',
                        secondary: '#a855f7',
                        accent: '#06b6d4',
                        dark: '#0f172a',
                        'dark-secondary': '#1e293b',
                        'dark-accent': '#334155'
                    }
                }
            }
        }
    </script>
    
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    
    <!-- Highlight.js for code highlighting -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/highlight.min.js"></script>
    
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        .card-bg {
            background: rgba(30, 41, 59, 0.6);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(148, 163, 184, 0.1);
        }
        
        .tag {
            background: rgba(99, 102, 241, 0.2);
            border: 1px solid rgba(99, 102, 241, 0.3);
            color: #c4b5fd;
        }
        
        .hero-gradient {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .nav-link {
            position: relative;
            transition: all 0.3s ease;
        }
        
        .nav-link::after {
            content: '';
            position: absolute;
            bottom: -4px;
            left: 0;
            width: 0;
            height: 2px;
            background: linear-gradient(90deg, #6366f1, #8b5cf6);
            transition: width 0.3s ease;
        }
        
        .nav-link:hover::after {
            width: 100%;
        }
        
        .project-card {
            transition: all 0.3s ease;
        }
        
        .project-card:hover {
            box-shadow: 0 20px 40px rgba(99, 102, 241, 0.2);
        }
        
        .modal-content {
            max-height: 90vh;
            overflow-y: auto;
        }
        
        .modal-content::-webkit-scrollbar {
            width: 8px;
        }
        
        .modal-content::-webkit-scrollbar-track {
            background: rgba(30, 41, 59, 0.5);
            border-radius: 4px;
        }
        
        .modal-content::-webkit-scrollbar-thumb {
            background: rgba(99, 102, 241, 0.5);
            border-radius: 4px;
        }
        
        .modal-content::-webkit-scrollbar-thumb:hover {
            background: rgba(99, 102, 241, 0.7);
        }
        
        .thumbnail {
            transition: all 0.3s ease;
            cursor: pointer;
            border: 2px solid transparent;
        }
        
        .thumbnail:hover, .thumbnail.active {
            border-color: #6366f1;
            transform: scale(1.05);
        }
        
        .prose-custom {
            line-height: 1.7;
        }
        
        .prose-custom h1, .prose-custom h2, .prose-custom h3 {
            color: #e2e8f0;
            font-weight: 700;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        
        .prose-custom p {
            margin-bottom: 1.5rem;
        }
        
        .prose-custom ul, .prose-custom ol {
            margin-bottom: 1.5rem;
            padding-left: 1.5rem;
        }
        
        .prose-custom li {
            margin-bottom: 0.5rem;
        }
        
        .prose-custom blockquote {
            border-left: 4px solid #6366f1;
            padding-left: 1rem;
            margin: 1.5rem 0;
            font-style: italic;
            color: #cbd5e1;
        }
        
        .prose-custom pre {
            background: rgba(15, 23, 42, 0.8);
            padding: 1rem;
            border-radius: 0.5rem;
            overflow-x: auto;
            margin: 1.5rem 0;
        }
        
        .prose-custom code {
            background: rgba(99, 102, 241, 0.1);
            padding: 0.2rem 0.4rem;
            border-radius: 0.25rem;
            font-size: 0.9em;
        }
        
        .prose-custom pre code {
            background: none;
            padding: 0;
        }
        
        .prose-custom strong {
            color: #f1f5f9;
            font-weight: 600;
        }
        
        .blog-content-truncated {
            max-height: 200px;
            overflow: hidden;
            position: relative;
        }
        
        .blog-content-truncated::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 60px;
            background: linear-gradient(transparent, rgba(30, 41, 59, 0.9));
        }
        
        .blog-content-expanded {
            max-height: none;
        }
        
        .blog-tag-filter {
            transition: all 0.3s ease;
        }
        
        .blog-tag-filter:hover {
            background: rgba(99, 102, 241, 0.3);
        }
        
        .blog-tag-filter.active {
            background: #6366f1;
            color: white;
        }
        
        .poll-option {
            transition: all 0.3s ease;
        }
        
        .poll-option:hover:not(:disabled) {
            background: rgba(99, 102, 241, 0.2);
        }
        
        #bg-canvas {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            opacity: 0.5;
        }
        
        .interactive-cover-container {
            height: 200px;
            overflow: hidden;
            position: relative;
        }
        
        .interactive-cover-container svg {
            width: 100%;
            height: 100%;
        }
        
        .svg-hidden {
            opacity: 0;
            transition: opacity 0.5s ease;
        }
        
        .project-card:hover .svg-hidden {
            opacity: 1;
        }
        
        .draw-on-hover {
            stroke-dasharray: 1000;
            stroke-dashoffset: 1000;
            transition: stroke-dashoffset 1s ease;
        }
        
        .project-card:hover .draw-on-hover {
            stroke-dashoffset: 0;
        }
        
        .pm-gear {
            animation: rotate 4s linear infinite;
        }
        
        @keyframes rotate {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        
        .churn-dot {
            animation: pulse 2s infinite;
        }
        
        .churn-dot-synthetic {
            animation: pulse 2s infinite 0.5s;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .toolkit-glow {
            animation: glow 3s ease-in-out infinite alternate;
        }
        
        @keyframes glow {
            from { filter: drop-shadow(0 0 5px rgba(139, 92, 246, 0.5)); }
            to { filter: drop-shadow(0 0 20px rgba(139, 92, 246, 0.8)); }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: scale(0.9); }
            to { opacity: 1; transform: scale(1); }
        }
        
        @keyframes fadeOut {
            from { opacity: 1; transform: scale(1); }
            to { opacity: 0; transform: scale(0.9); }
        }
        
        .fade-in {
            animation: fadeIn 0.3s ease-out;
        }
        
        @media (max-width: 768px) {
            .modal-content {
                margin: 1rem;
                max-height: 95vh;
            }
            
            .interactive-cover-container {
                height: 150px;
            }
        }
    </style>
</head>
<body class="bg-gray-900 text-white">
    <canvas id="bg-canvas"></canvas>
    
    <!-- Navigation -->
    <nav class="fixed top-0 left-0 right-0 z-50 bg-dark/80 backdrop-blur-sm border-b border-gray-800">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex justify-between items-center h-16">
                <div class="flex-shrink-0">
                    <span class="text-2xl font-bold hero-gradient">AD</span>
                </div>
                <div class="hidden md:block">
                    <div class="ml-10 flex items-baseline space-x-8">
                        <a href="#home" class="nav-link text-gray-300 hover:text-white px-3 py-2 text-sm font-medium">Home</a>
                        <a href="#about" class="nav-link text-gray-300 hover:text-white px-3 py-2 text-sm font-medium">About</a>
                        <a href="#projects" class="nav-link text-gray-300 hover:text-white px-3 py-2 text-sm font-medium">Projects</a>
                        <a href="#playground" class="nav-link text-gray-300 hover:text-white px-3 py-2 text-sm font-medium">Playground</a>
                        <a href="#skills" class="nav-link text-gray-300 hover:text-white px-3 py-2 text-sm font-medium">Skills</a>
                        <a href="#blog" class="nav-link text-gray-300 hover:text-white px-3 py-2 text-sm font-medium">Blog</a>
                        <a href="#contact" class="nav-link text-gray-300 hover:text-white px-3 py-2 text-sm font-medium">Contact</a>
                    </div>
                </div>
                <div class="md:hidden">
                    <button id="mobile-menu-button" class="text-gray-300 hover:text-white">
                        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"></path>
                        </svg>
                    </button>
                </div>
            </div>
        </div>
        <div id="mobile-menu" class="md:hidden hidden">
            <div class="px-2 pt-2 pb-3 space-y-1 sm:px-3 bg-dark/90 backdrop-blur-sm">
                <a href="#home" class="block px-3 py-2 text-base font-medium text-gray-300 hover:text-white">Home</a>
                <a href="#about" class="block px-3 py-2 text-base font-medium text-gray-300 hover:text-white">About</a>
                <a href="#projects" class="block px-3 py-2 text-base font-medium text-gray-300 hover:text-white">Projects</a>
                <a href="#playground" class="block px-3 py-2 text-base font-medium text-gray-300 hover:text-white">Playground</a>
                <a href="#skills" class="block px-3 py-2 text-base font-medium text-gray-300 hover:text-white">Skills</a>
                <a href="#blog" class="block px-3 py-2 text-base font-medium text-gray-300 hover:text-white">Blog</a>
                <a href="#contact" class="block px-3 py-2 text-base font-medium text-gray-300 hover:text-white">Contact</a>
            </div>
        </div>
    </nav>

    <!-- Hero Section -->
    <section id="home" class="min-h-screen flex items-center justify-center text-center px-4 pt-16">
        <div class="max-w-4xl mx-auto">
            <div class="mb-8">
                <img src="https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/profile.jpg" 
                     alt="Adarsh Divase" 
                     class="w-48 h-48 rounded-full mx-auto mb-6 border-4 border-indigo-500 shadow-2xl"
                     onerror="this.onerror=null;this.src='https://ui-avatars.com/api/?name=Adarsh+Divase&background=6366f1&color=fff&size=192';">
            </div>
            <h1 class="text-5xl md:text-7xl font-bold mb-6">
                Hi, I'm <span class="hero-gradient">Adarsh Divase</span>
            </h1>
            <p class="text-xl md:text-2xl text-gray-300 mb-8">
                <span id="role-text" class="hero-gradient font-semibold"></span>
            </p>
            <p class="text-lg text-gray-400 max-w-2xl mx-auto mb-8">
                Passionate about transforming data into intelligent solutions. Currently pursuing B.Tech in Computer Science, 
                specializing in AI/ML, Deep Learning, and scalable system architecture.
            </p>
            <div class="flex flex-col sm:flex-row gap-4 justify-center">
                <a href="#projects" class="bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-3 px-8 rounded-lg transition-colors duration-300">
                    View My Work
                </a>
                <a href="#contact" class="border border-indigo-600 text-indigo-400 hover:bg-indigo-600 hover:text-white font-medium py-3 px-8 rounded-lg transition-colors duration-300">
                    Get In Touch
                </a>
            </div>
        </div>
    </section>

    <!-- About Section -->
    <section id="about" class="py-20 px-4">
        <div class="max-w-6xl mx-auto">
            <div class="text-center mb-16">
                <h2 class="text-4xl font-bold mb-4">About Me</h2>
                <p class="text-xl text-gray-400">Passionate about AI, Data Science, and Building Scalable Solutions</p>
            </div>
            
            <div class="grid lg:grid-cols-2 gap-12 items-center">
                <div class="space-y-6">
                    <div class="card-bg p-8 rounded-2xl">
                        <h3 class="text-2xl font-bold mb-4 text-indigo-400">🎓 Education</h3>
                        <p class="text-gray-300 mb-2"><strong>B.Tech in Computer Science Engineering</strong></p>
                        <p class="text-gray-400">A. C. Patil College of Engineering, Mumbai University</p>
                        <p class="text-gray-400">Expected Graduation: 2025</p>
                    </div>
                    
                    <div class="card-bg p-8 rounded-2xl">
                        <h3 class="text-2xl font-bold mb-4 text-indigo-400">💼 Experience</h3>
                        <p class="text-gray-300 mb-2"><strong>Python Backend Developer Intern</strong></p>
                        <p class="text-gray-400">Developed scalable microservices and APIs, improved performance by 25%, and managed PostgreSQL databases.</p>
                    </div>
                </div>
                
                <div class="space-y-6">
                    <div class="card-bg p-8 rounded-2xl">
                        <h3 class="text-2xl font-bold mb-4 text-indigo-400">🏆 Achievements</h3>
                        <ul class="text-gray-300 space-y-2">
                            <li>• Deployed 5+ production ML models with 99.9% uptime</li>
                            <li>• Improved API response times by 25% through optimization</li>
                            <li>• Built real-time data pipelines processing 10,000+ data points/minute</li>
                            <li>• Achieved 92% accuracy in sales forecasting models</li>
                        </ul>
                    </div>
                    
                    <div class="card-bg p-8 rounded-2xl">
                        <h3 class="text-2xl font-bold mb-4 text-indigo-400">🎯 Focus Areas</h3>
                        <div class="flex flex-wrap gap-2">
                            <span class="tag rounded-lg px-3 py-1">Machine Learning</span>
                            <span class="tag rounded-lg px-3 py-1">Deep Learning</span>
                            <span class="tag rounded-lg px-3 py-1">MLOps</span>
                            <span class="tag rounded-lg px-3 py-1">Data Engineering</span>
                            <span class="tag rounded-lg px-3 py-1">Real-time Systems</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Projects Section -->
    <section id="projects" class="py-20 px-4">
        <div class="max-w-7xl mx-auto">
            <div class="text-center mb-16">
                <h2 class="text-4xl font-bold mb-4">Featured Projects</h2>
                <p class="text-xl text-gray-400">A showcase of my technical expertise and innovative solutions</p>
            </div>
            
            <div id="projects-grid" class="grid md:grid-cols-2 xl:grid-cols-3 gap-8">
                <!-- Projects will be populated by JavaScript -->
            </div>
        </div>
    </section>

    <!-- Playground Section -->
    <section id="playground" class="py-20 px-4 bg-gray-900/50">
        <div class="max-w-7xl mx-auto">
            <div class="text-center mb-16">
                <h2 class="text-4xl font-bold mb-4">Interactive Playground</h2>
                <p class="text-xl text-gray-400">Try out my live applications and demos</p>
            </div>
            
            <div id="playground-apps-grid" class="grid md:grid-cols-2 xl:grid-cols-3 gap-8">
                <!-- Playground apps will be populated by JavaScript -->
            </div>
        </div>
    </section>

    <!-- Skills Section -->
    <section id="skills" class="py-20 px-4">
        <div class="max-w-7xl mx-auto">
            <div class="text-center mb-16">
                <h2 class="text-4xl font-bold mb-4">Technical Skills</h2>
                <p class="text-xl text-gray-400">Technologies and tools I work with</p>
            </div>
            
            <div class="grid md:grid-cols-2 xl:grid-cols-3 gap-8">
                <!-- Skills will be populated by JavaScript -->
            </div>
        </div>
    </section>

    <!-- Blog Section -->
    <section id="blog" class="py-20 px-4 bg-gray-900/50">
        <div class="max-w-7xl mx-auto">
            <div class="text-center mb-16">
                <h2 class="text-4xl font-bold mb-4">Latest Blog Posts</h2>
                <p class="text-xl text-gray-400">Insights, tutorials, and thoughts on AI and technology</p>
            </div>
            
            <div id="blog-filters" class="flex flex-wrap gap-2 justify-center mb-12">
                <!-- Blog filters will be populated by JavaScript -->
            </div>
            
            <div id="blog-posts-container" class="space-y-12">
                <!-- Blog posts will be populated by JavaScript -->
            </div>
        </div>
    </section>

    <!-- Contact Section -->
    <section id="contact" class="py-20 px-4">
        <div class="max-w-4xl mx-auto">
            <div class="text-center mb-16">
                <h2 class="text-4xl font-bold mb-4">Get In Touch</h2>
                <p class="text-xl text-gray-400">Let's discuss opportunities and collaborate on exciting projects</p>
            </div>
            
            <div class="grid md:grid-cols-2 gap-12">
                <div class="space-y-8">
                    <div class="card-bg p-8 rounded-2xl">
                        <h3 class="text-2xl font-bold mb-6 text-indigo-400">Connect With Me</h3>
                        <div class="space-y-4">
                            <a href="mailto:adarshdivase@gmail.com" class="flex items-center text-gray-300 hover:text-indigo-400 transition-colors">
                                <svg class="w-6 h-6 mr-3" fill="currentColor" viewBox="0 0 24 24">
                                    <path d="M24 4.557c-.883.392-1.832.656-2.828.775 1.017-.609 1.798-1.574 2.165-2.724-.951.564-2.005.974-3.127 1.195-.897-.957-2.178-1.555-3.594-1.555-3.179 0-5.515 2.966-4.797 6.045-4.091-.205-7.719-2.165-10.148-5.144-1.29 2.213-.669 5.108 1.523 6.574-.806-.026-1.566-.247-2.229-.616-.054 2.281 1.581 4.415 3.949 4.89-.693.188-1.452.232-2.224.084.626 1.956 2.444 3.379 4.6 3.419-2.07 1.623-4.678 2.348-7.29 2.04 2.179 1.397 4.768 2.212 7.548 2.212 9.142 0 14.307-7.721 13.995-14.646.962-.695 1.797-1.562 2.457-2.549z"/>
                                </svg>
                                LinkedIn Profile
                            </a>
                            <a href="https://github.com/adarshdivase" class="flex items-center text-gray-300 hover:text-indigo-400 transition-colors">
                                <svg class="w-6 h-6 mr-3" fill="currentColor" viewBox="0 0 24 24">
                                    <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                                </svg>
                                GitHub Profile
                            </a>
                            <p class="flex items-center text-gray-300">
                                <svg class="w-6 h-6 mr-3" fill="currentColor" viewBox="0 0 24 24">
                                    <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/>
                                </svg>
                                Mumbai, India
                            </p>
                        </div>
                    </div>
                    
                    <div class="card-bg p-8 rounded-2xl">
                        <h3 class="text-2xl font-bold mb-6 text-indigo-400">Recent Posts</h3>
                        <div id="recent-posts-container" class="space-y-4">
                            <!-- Recent posts will be populated by JavaScript -->
                        </div>
                    </div>
                </div>
                
                <div class="space-y-8">
                    <div class="card-bg p-8 rounded-2xl">
                        <h3 class="text-2xl font-bold mb-6 text-indigo-400">Ask Me Anything</h3>
                        <form id="comment-form" class="space-y-4">
                            <div>
                                <label class="block text-sm font-medium text-gray-300 mb-2">Your Name</label>
                                <input type="text" required class="w-full px-4 py-3 rounded-lg bg-gray-800 border border-gray-700 text-white focus:outline-none focus:border-indigo-500">
                            </div>
                            <div>
                                <label class="block text-sm font-medium text-gray-300 mb-2">Email</label>
                                <input type="email" required class="w-full px-4 py-3 rounded-lg bg-gray-800 border border-gray-700 text-white focus:outline-none focus:border-indigo-500">
                            </div>
                            <div>
                                <label class="block text-sm font-medium text-gray-300 mb-2">Your Question</label>
                                <textarea required rows="4" class="w-full px-4 py-3 rounded-lg bg-gray-800 border border-gray-700 text-white focus:outline-none focus:border-indigo-500"></textarea>
                            </div>
                            <button type="submit" class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-3 px-6 rounded-lg transition-colors duration-300">
                                Send Message
                            </button>
                        </form>
                    </div>
                    
                    <div class="card-bg p-8 rounded-2xl">
                        <h3 class="text-2xl font-bold mb-6 text-indigo-400">Quick Poll</h3>
                        <p class="text-gray-300 mb-4">What type of AI content would you like to see more of?</p>
                        <div class="space-y-3">
                            <button class="poll-option w-full text-left px-4 py-3 rounded-lg bg-gray-800 border border-gray-700 hover:bg-gray-700 transition-colors">
                                🤖 Machine Learning Tutorials
                            </button>