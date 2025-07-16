import streamlit as st
import pandas as pd
from auth import AuthManager
from hr_analytics import HRAnalytics
from utils import load_employee_data

# Set page configuration
st.set_page_config(
    page_title="🚀 Human Resources Analysis Platform",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_data' not in st.session_state:
    st.session_state.user_data = None
if 'employee_df' not in st.session_state:
    st.session_state.employee_df = None

# Load employee data
@st.cache_data
def get_employee_data():
    return load_employee_data()

# Initialize authentication manager
auth_manager = AuthManager()

def main():
    # Load employee data
    if st.session_state.employee_df is None:
        st.session_state.employee_df = get_employee_data()
    
    # Check authentication
    if not st.session_state.authenticated:
        show_login_page()
    else:
        show_main_application()

def show_login_page():
    # Ultra-modern CSS for advanced login page
    st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #6B73FF 100%);
            color: white;
            min-height: 100vh;
            position: relative;
        }
        
        /* Animated background particles */
        .main::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                radial-gradient(circle at 20% 30%, rgba(255, 255, 255, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 80% 70%, rgba(255, 255, 255, 0.05) 0%, transparent 50%),
                radial-gradient(circle at 40% 80%, rgba(79, 172, 254, 0.1) 0%, transparent 50%);
            animation: float 20s ease-in-out infinite;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-20px); }
        }
        
        .login-container {
            max-width: 480px;
            margin: 0 auto;
            padding: 60px 40px;
            background: rgba(255, 255, 255, 0.18);
            border-radius: 30px;
            border: 3px solid rgba(255, 255, 255, 0.4);
            box-shadow: 
                0 25px 50px rgba(0, 0, 0, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.3);
            backdrop-filter: blur(20px);
            margin-top: 3vh;
            animation: slideIn 1s ease-out;
            position: relative;
            overflow: hidden;
        }
        
        .login-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            animation: shimmer 3s infinite;
        }
        
        @keyframes shimmer {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(50px) scale(0.9); }
            to { opacity: 1; transform: translateY(0) scale(1); }
        }
        
        .login-header {
            text-align: center;
            font-size: 3.5rem;
            font-weight: 800;
            margin-bottom: 10px;
            color: white;
            text-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
            background: linear-gradient(45deg, #fff, #4facfe);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: glow 2s ease-in-out infinite alternate;
        }
        
        @keyframes glow {
            from { text-shadow: 0 0 10px rgba(79, 172, 254, 0.5); }
            to { text-shadow: 0 0 20px rgba(79, 172, 254, 0.8); }
        }
        
        .login-subtitle {
            text-align: center;
            font-size: 1.3rem;
            margin-bottom: 40px;
            color: rgba(255, 255, 255, 0.95);
            font-weight: 300;
            letter-spacing: 0.5px;
        }
        
        .security-badge {
            display: block;
            text-align: center;
            background: linear-gradient(45deg, #ff6b6b, #ffa726, #4facfe);
            color: white;
            padding: 12px 24px;
            border-radius: 25px;
            font-size: 1rem;
            font-weight: bold;
            margin: 0 auto 25px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
            animation: pulse 2s infinite;
            max-width: 250px;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .feature-icons {
            display: flex;
            justify-content: space-around;
            margin: 35px 0;
            padding: 20px 0;
            border-top: 1px solid rgba(255, 255, 255, 0.2);
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .feature-icon {
            text-align: center;
            color: rgba(255, 255, 255, 0.9);
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .feature-icon:hover {
            transform: translateY(-5px);
            color: #4facfe;
        }
        
        .feature-icon-emoji {
            font-size: 2.5rem;
            margin-bottom: 8px;
            display: block;
            animation: bounce 2s infinite;
        }
        
        .feature-icon-text {
            font-size: 0.9rem;
            font-weight: 500;
        }
        
        @keyframes bounce {
            0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
            40% { transform: translateY(-10px); }
            60% { transform: translateY(-5px); }
        }
        
        .stTextInput > div > div > input {
            background: rgba(255, 255, 255, 0.25);
            color: white;
            border: 2px solid rgba(255, 255, 255, 0.4);
            border-radius: 20px;
            padding: 18px 24px;
            font-size: 1.1rem;
            transition: all 0.4s ease;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .stTextInput > div > div > input:focus {
            background: rgba(255, 255, 255, 0.35);
            border-color: #4facfe;
            box-shadow: 0 0 30px rgba(79, 172, 254, 0.6);
            transform: translateY(-2px);
        }
        
        .stTextInput > div > div > input::placeholder {
            color: rgba(255, 255, 255, 0.8);
            font-weight: 500;
        }
        
        .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 50%, #4facfe 100%);
            color: white;
            border: none;
            border-radius: 20px;
            padding: 18px;
            font-weight: bold;
            font-size: 1.2rem;
            transition: all 0.4s ease;
            box-shadow: 0 8px 25px rgba(79, 172, 254, 0.5);
            position: relative;
            overflow: hidden;
        }
        
        .stButton > button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
            transition: left 0.6s;
        }
        
        .stButton > button:hover::before {
            left: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 35px rgba(79, 172, 254, 0.7);
            background: linear-gradient(135deg, #00f2fe 0%, #4facfe 50%, #00f2fe 100%);
        }
        
        .demo-info {
            background: rgba(255, 255, 255, 0.15);
            padding: 25px;
            border-radius: 20px;
            margin-top: 35px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(10px);
        }
        
        .demo-info h4 {
            color: #4facfe;
            margin-bottom: 15px;
            font-size: 1.2rem;
            text-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
        }
        
        .demo-credentials {
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 12px;
            margin: 10px 0;
            border-left: 4px solid #4facfe;
            transition: all 0.3s ease;
        }
        
        .demo-credentials:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: translateX(5px);
        }
        
        .floating-shapes {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: -1;
        }
        
        .shape {
            position: absolute;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            animation: float-shapes 15s infinite linear;
        }
        
        .shape:nth-child(1) {
            width: 80px;
            height: 80px;
            left: 10%;
            animation-delay: 0s;
        }
        
        .shape:nth-child(2) {
            width: 120px;
            height: 120px;
            left: 80%;
            animation-delay: -5s;
        }
        
        .shape:nth-child(3) {
            width: 60px;
            height: 60px;
            left: 50%;
            animation-delay: -10s;
        }
        
        @keyframes float-shapes {
            0% { transform: translateY(100vh) rotate(0deg); }
            100% { transform: translateY(-100px) rotate(360deg); }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Add floating shapes background
    st.markdown("""
    <div class="floating-shapes">
        <div class="shape"></div>
        <div class="shape"></div>
        <div class="shape"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        # Security badge and enhanced header
        st.markdown('<div class="security-badge">🔒 Secure Employee Portal</div>', unsafe_allow_html=True)
        st.markdown('<h1 class="login-header">🚀 HR Analytics</h1>', unsafe_allow_html=True)
        st.markdown('<p class="login-subtitle">Access Your Workforce Intelligence Platform</p>', unsafe_allow_html=True)
        
        # Feature icons with enhanced styling
        st.markdown("""
        <div class="feature-icons">
            <div class="feature-icon">
                <span class="feature-icon-emoji">📊</span>
                <div class="feature-icon-text">Advanced Analytics</div>
            </div>
            <div class="feature-icon">
                <span class="feature-icon-emoji">🤖</span>
                <div class="feature-icon-text">AI Predictions</div>
            </div>
            <div class="feature-icon">
                <span class="feature-icon-emoji">👥</span>
                <div class="feature-icon-text">Team Insights</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit_button = st.form_submit_button("🔐 Sign In")
            
            if submit_button:
                if username and password:
                    user_data = auth_manager.authenticate(username, password, st.session_state.employee_df)
                    if user_data is not None:
                        st.session_state.authenticated = True
                        st.session_state.user_data = user_data
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.warning("Please enter both username and password")
        
        # Login footer info
        st.markdown("""
        <div style="text-align: center; margin-top: 30px; font-size: 0.9rem; opacity: 0.8;">
            <p>Secure access to your HR analytics platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display demo credentials info
        st.markdown("""
        <div style="text-align: center; margin-top: 20px; font-size: 0.9rem; opacity: 0.8;">
            <p>Use any employee's Username and Password from the system</p>
            <p>Example: Username: "Ira Nair", Password: "2981"</p>
        </div>
        """, unsafe_allow_html=True)

def show_main_application():
    # Custom CSS for main application
    st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .stTabs [data-baseweb="tab-list"] {
            display: flex;
            justify-content: center;
            gap: 24px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            padding: 10px 20px;
            color: white;
            font-weight: bold;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        }
        .metric-card {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
            padding: 20px;
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
        }
        .header-title {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
            color: white;
        }
        .section-header {
            font-size: 1.5rem;
            font-weight: bold;
            margin: 20px 0;
            color: #4facfe;
        }
        .logout-button {
            background: linear-gradient(90deg, #ff6b6b 0%, #ff8e8e 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: bold;
            cursor: pointer;
        }
        
        /* Center all page titles */
        .centered-title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: bold;
            margin: 20px 0;
            color: #4facfe;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with logout button in top-left corner
    col1, col2, col3 = st.columns([1, 6, 1])
    
    with col1:
        # Logout button in top-left corner
        if st.button("🚪 Logout", key="logout_btn", help="Sign out of the application"):
            auth_manager.logout()
            st.rerun()
    
    with col2:
        # Centered main title
        st.markdown('<h1 class="header-title">🚀 Human Resources Analysis Platform</h1>', unsafe_allow_html=True)
        st.markdown(f'<p style="text-align: center; font-size: 1.2rem; margin-bottom: 30px;">Welcome, {st.session_state.user_data["Username"]} ({st.session_state.user_data["department"]})</p>', unsafe_allow_html=True)
    
    with col3:
        # Empty column for balance
        st.empty()
    
    # Initialize HR Analytics
    hr_analytics = HRAnalytics(st.session_state.employee_df, st.session_state.user_data)
    
    # Show main dashboard
    hr_analytics.show_dashboard()

if __name__ == "__main__":
    main()
