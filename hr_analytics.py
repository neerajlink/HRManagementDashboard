import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class HRAnalytics:
    def __init__(self, employee_df, user_data):
        self.employee_df = employee_df
        self.user_data = user_data
        self.permissions = self.get_user_permissions()
        
    def get_user_permissions(self):
        """Get user permissions based on role"""
        position = self.user_data.get('position', '')
        
        if position == 'Director':
            return {
                'view_dashboard': True,
                'view_analytics': True,
                'manage_employees': True,
                'view_reports': True,
                'ml_predictions': True
            }
        elif position == 'Manager':
            return {
                'view_dashboard': True,
                'view_analytics': True,
                'manage_employees': True,
                'view_reports': True,
                'ml_predictions': False
            }
        else:
            return {
                'view_dashboard': True,
                'view_analytics': False,
                'manage_employees': False,
                'view_reports': False,
                'ml_predictions': False
            }
    
    def show_dashboard(self):
        """Main dashboard interface"""
        # Create tabs based on permissions
        tabs = ["📊 Dashboard"]
        
        if self.permissions['view_analytics']:
            tabs.append("📈 Analytics")
        
        if self.permissions['ml_predictions']:
            tabs.append("🤖 ML Prediction")
        
        if self.permissions['manage_employees']:
            tabs.append("👥 Employee Management")
        
        if self.permissions['view_reports']:
            tabs.append("📋 Reports")
        
        tab_objects = st.tabs(tabs)
        
        # Dashboard tab
        with tab_objects[0]:
            self.show_dashboard_tab()
        
        # Analytics tab
        if self.permissions['view_analytics'] and len(tab_objects) > 1:
            with tab_objects[1]:
                self.show_analytics_tab()
        
        # ML Prediction tab
        if self.permissions['ml_predictions']:
            tab_index = 2 if self.permissions['view_analytics'] else 1
            if len(tab_objects) > tab_index:
                with tab_objects[tab_index]:
                    self.show_ml_prediction_tab()
        
        # Employee Management tab
        if self.permissions['manage_employees']:
            tab_index = len(tabs) - 2 if self.permissions['view_reports'] else len(tabs) - 1
            if len(tab_objects) > tab_index:
                with tab_objects[tab_index]:
                    self.show_employee_management_tab()
        
        # Reports tab
        if self.permissions['view_reports']:
            with tab_objects[-1]:
                self.show_reports_tab()
    
    def show_dashboard_tab(self):
        """Dashboard with key metrics"""
        st.markdown('<h2 class="centered-title">📊 Dashboard</h2>', unsafe_allow_html=True)
        
        # Filter data based on user department if not Director
        if self.user_data.get('position') != 'Director':
            filtered_df = self.employee_df[self.employee_df['department'] == self.user_data.get('department')]
        else:
            filtered_df = self.employee_df
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Employees</h3>
                <h2 style="color: #4facfe;">{len(filtered_df)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Calculate attrition rate from risk categories
            high_risk = len(filtered_df[filtered_df['risk_category'] == 'High'])
            critical_risk = len(filtered_df[filtered_df['risk_category'] == 'Critical'])
            attrition_rate = ((high_risk + critical_risk) / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>High Risk Rate</h3>
                <h2 style="color: #ff6b6b;">{attrition_rate:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_salary = filtered_df['monthly_income'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>Average Salary</h3>
                <h2 style="color: #4ecdc4;">${avg_salary:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_tenure = filtered_df['tenure'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>Average Tenure</h3>
                <h2 style="color: #45b7d1;">{avg_tenure:.1f} years</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Department distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="section-header">Department Distribution</h3>', unsafe_allow_html=True)
            dept_counts = filtered_df['department'].value_counts()
            fig = px.pie(values=dept_counts.values, names=dept_counts.index,
                        title="Employee Distribution by Department")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<h3 class="section-header">Risk Category Distribution</h3>', unsafe_allow_html=True)
            risk_counts = filtered_df['risk_category'].value_counts()
            colors = {'Low': '#4facfe', 'Medium': '#ffa726', 'High': '#ff6b6b', 'Critical': '#d32f2f'}
            
            fig = px.bar(x=risk_counts.index, y=risk_counts.values,
                        title="Employee Risk Distribution",
                        color=risk_counts.index,
                        color_discrete_map=colors)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
    
    def show_analytics_tab(self):
        """Advanced analytics dashboard"""
        st.markdown('<h2 class="centered-title">📈 Analytics</h2>', unsafe_allow_html=True)
        
        # Filter data based on user role
        if self.user_data.get('position') != 'Director':
            filtered_df = self.employee_df[self.employee_df['department'] == self.user_data.get('department')]
        else:
            filtered_df = self.employee_df
        
        # Performance vs Satisfaction Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="section-header">Performance vs Satisfaction</h3>', unsafe_allow_html=True)
            fig = px.scatter(filtered_df, x='performance_rating', y='job_satisfaction',
                           color='risk_category', title="Performance vs Job Satisfaction",
                           size='monthly_income', hover_data=['Username', 'department'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<h3 class="section-header">Salary vs Tenure</h3>', unsafe_allow_html=True)
            fig = px.scatter(filtered_df, x='tenure', y='monthly_income',
                           color='position', title="Salary vs Tenure Analysis",
                           hover_data=['Username', 'department'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        # Age and Position Analysis
        st.markdown('<h3 class="section-header">Age and Position Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            age_position = filtered_df.groupby(['position', 'age']).size().reset_index(name='count')
            fig = px.box(filtered_df, x='position', y='age', title="Age Distribution by Position")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(filtered_df, x='age', color='department', 
                             title="Age Distribution by Department", nbins=20)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
    
    def show_ml_prediction_tab(self):
        """Machine Learning predictions tab"""
        st.markdown('<h2 class="centered-title">🤖 ML Prediction</h2>', unsafe_allow_html=True)
        
        # Prepare data for ML (create synthetic attrition for demonstration)
        df_ml = self.employee_df.copy()
        
        # Create attrition based on risk categories
        df_ml['attrition'] = df_ml['risk_category'].map({
            'Low': 0, 'Medium': 0, 'High': 1, 'Critical': 1
        })
        
        # Prepare features
        le_dept = LabelEncoder()
        le_pos = LabelEncoder()
        
        df_ml['department_encoded'] = le_dept.fit_transform(df_ml['department'])
        df_ml['position_encoded'] = le_pos.fit_transform(df_ml['position'])
        
        features = ['age', 'tenure', 'monthly_income', 'performance_rating', 
                   'job_satisfaction', 'department_encoded', 'position_encoded']
        
        X = df_ml[features]
        y = df_ml['attrition']
        
        # Train models
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        
        # Create enhanced tabs for ML functionalities
        ml_tab1, ml_tab2, ml_tab3, ml_tab4 = st.tabs([
            "📊 Model Performance", 
            "🎯 Individual Risk Assessment", 
            "📈 Attrition Risk Analysis", 
            "⚠️ Advanced Risk Management"
        ])
        
        with ml_tab1:
            # Model Performance Tab
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<h3 class="section-header">Model Performance</h3>', unsafe_allow_html=True)
                st.write(f"**Random Forest Accuracy:** {rf_accuracy:.3f}")
                
                # Confusion Matrix
                cm = confusion_matrix(y_test, rf_pred)
                fig = px.imshow(cm, text_auto=True, title="Confusion Matrix",
                               color_continuous_scale='Blues')
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown('<h3 class="section-header">Feature Importance</h3>', unsafe_allow_html=True)
                
                # Feature importance
                importance = rf_model.feature_importances_
                feature_importance_df = pd.DataFrame({
                    'feature': features,
                    'importance': importance
                }).sort_values('importance', ascending=True)
                
                fig = px.bar(feature_importance_df, x='importance', y='feature',
                            orientation='h', title="Feature Importance",
                            color='importance', color_continuous_scale='Viridis')
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
        
        with ml_tab2:
            # Individual Risk Assessment Tab
            st.markdown('<h3 class="section-header">Individual Employee Risk Assessment</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**Employee Parameters:**")
                
                # Input fields for prediction
                pred_age = st.slider("Age", 22, 65, 35)
                pred_tenure = st.slider("Tenure (years)", 0, 20, 5)
                pred_income = st.slider("Monthly Income", 3000, 15000, 8000)
                pred_performance = st.slider("Performance Rating", 1, 5, 3)
                pred_satisfaction = st.slider("Job Satisfaction", 1, 5, 3)
                pred_dept = st.selectbox("Department", self.employee_df['department'].unique())
                pred_position = st.selectbox("Position", self.employee_df['position'].unique())
                
                # Encode categorical variables for prediction
                pred_dept_encoded = le_dept.transform([pred_dept])[0]
                pred_pos_encoded = le_pos.transform([pred_position])[0]
                
                # Create prediction input
                pred_input = np.array([[pred_age, pred_tenure, pred_income, pred_performance, 
                                      pred_satisfaction, pred_dept_encoded, pred_pos_encoded]])
                
                # Make prediction
                if st.button("🔮 Predict Attrition Risk"):
                    prediction = rf_model.predict(pred_input)[0]
                    prediction_proba = rf_model.predict_proba(pred_input)[0]
                    
                    # Store prediction in session state
                    st.session_state.prediction = prediction
                    st.session_state.prediction_proba = prediction_proba
            
            with col2:
                st.markdown("**Prediction Results:**")
                
                if hasattr(st.session_state, 'prediction'):
                    prediction = st.session_state.prediction
                    prediction_proba = st.session_state.prediction_proba
                    
                    # Display prediction
                    risk_level = "High Risk" if prediction == 1 else "Low Risk"
                    risk_color = "#ff6b6b" if prediction == 1 else "#4facfe"
                    
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; margin: 20px 0;">
                        <h4 style="color: {risk_color};">🎯 Attrition Risk: {risk_level}</h4>
                        <p><strong>Probability of Staying:</strong> {prediction_proba[0]:.2%}</p>
                        <p><strong>Probability of Leaving:</strong> {prediction_proba[1]:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Risk factors gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prediction_proba[1] * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Attrition Risk %"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 25], 'color': "lightgreen"},
                                {'range': [25, 50], 'color': "yellow"},
                                {'range': [50, 75], 'color': "orange"},
                                {'range': [75, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendations
                    st.markdown("**💡 Recommendations:**")
                    if prediction == 1:
                        st.markdown("""
                        - 🔍 **Schedule one-on-one meetings** to understand concerns
                        - 📈 **Provide career development opportunities**
                        - 💰 **Review compensation and benefits**
                        - 🎯 **Assign challenging projects** to increase engagement
                        - 👥 **Improve team dynamics** and work environment
                        """)
                    else:
                        st.markdown("""
                        - ✅ **Employee shows low attrition risk**
                        - 🌟 **Continue current engagement strategies**
                        - 📊 **Monitor performance regularly**
                        - 🚀 **Consider for leadership development**
                        """)
                else:
                    st.info("👆 Click 'Predict Attrition Risk' to see results")
        
        with ml_tab3:
            # Attrition Risk Analysis Tab
            st.markdown('<h3 class="section-header">Attrition Risk Analysis by Department</h3>', unsafe_allow_html=True)
            
            # Filter data based on user role
            if self.user_data.get('position') != 'Director':
                filtered_df = self.employee_df[self.employee_df['department'] == self.user_data.get('department')]
            else:
                filtered_df = self.employee_df
            
            # Department-wise attrition risk
            dept_risk = filtered_df.groupby('department').agg({
                'risk_category': lambda x: (x.isin(['High', 'Critical'])).sum(),
                'Employee_id': 'count'
            }).reset_index()
            dept_risk.columns = ['department', 'high_risk_count', 'total_employees']
            dept_risk['risk_percentage'] = (dept_risk['high_risk_count'] / dept_risk['total_employees']) * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(dept_risk, x='department', y='risk_percentage',
                            title="Attrition Risk by Department (%)",
                            color='risk_percentage',
                            color_continuous_scale='Reds')
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # High-risk employees list
                high_risk_employees = filtered_df[filtered_df['risk_category'].isin(['High', 'Critical'])]
                
                st.markdown("**🚨 High-Risk Employees:**")
                if len(high_risk_employees) > 0:
                    for _, employee in high_risk_employees.head(10).iterrows():
                        risk_color = "#d32f2f" if employee['risk_category'] == 'Critical' else "#ff6b6b"
                        st.markdown(f"""
                        <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px; margin: 5px 0; border-left: 4px solid {risk_color};">
                            <strong>{employee['Username']}</strong> - {employee['department']}<br>
                            <small>Risk: {employee['risk_category']} | Satisfaction: {employee['job_satisfaction']}/5</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("No high-risk employees found!")
            
            # Attrition trends over time (simulated)
            st.markdown('<h3 class="section-header">Risk Trends Analysis</h3>', unsafe_allow_html=True)
            
            # Create simulated trend data
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
            low_risk = [60, 65, 62, 68, 70, 72]
            medium_risk = [25, 22, 28, 20, 18, 16]
            high_risk = [15, 13, 10, 12, 12, 12]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=months, y=low_risk, mode='lines+markers', name='Low Risk', line=dict(color='#4facfe')))
            fig.add_trace(go.Scatter(x=months, y=medium_risk, mode='lines+markers', name='Medium Risk', line=dict(color='#ffa726')))
            fig.add_trace(go.Scatter(x=months, y=high_risk, mode='lines+markers', name='High Risk', line=dict(color='#ff6b6b')))
            
            fig.update_layout(
                title="Employee Risk Distribution Trends",
                xaxis_title="Month",
                yaxis_title="Percentage of Employees",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with ml_tab4:
            # Advanced Risk Management Tab
            st.markdown('<h3 class="section-header">Advanced Risk Management & Intervention</h3>', unsafe_allow_html=True)
            
            # Risk scoring methodology
            st.markdown("**📊 Risk Scoring Methodology:**")
            st.markdown("""
            Our attrition risk model evaluates multiple factors:
            - **Job Satisfaction** (40% weight): Lower satisfaction increases risk
            - **Performance Rating** (30% weight): Poor performance may indicate disengagement
            - **Department** (15% weight): Some departments have higher natural turnover
            - **Position Level** (10% weight): Career advancement opportunities affect retention
            - **Tenure & Age** (5% weight): Demographic factors influence stability
            """)
            
            # Interactive risk calculator
            st.markdown("**🔧 Interactive Risk Calculator:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                job_sat_weight = st.slider("Job Satisfaction Weight", 0.1, 0.7, 0.4)
                perf_weight = st.slider("Performance Weight", 0.1, 0.5, 0.3)
            
            with col2:
                dept_weight = st.slider("Department Weight", 0.05, 0.3, 0.15)
                pos_weight = st.slider("Position Weight", 0.05, 0.2, 0.1)
            
            with col3:
                demo_weight = st.slider("Demographics Weight", 0.01, 0.1, 0.05)
                threshold = st.slider("Risk Threshold", 0.3, 0.8, 0.5)
            
            # Calculate custom risk scores
            if st.button("🔄 Recalculate Risk Scores"):
                # Filter data based on user role
                if self.user_data.get('position') != 'Director':
                    calc_df = self.employee_df[self.employee_df['department'] == self.user_data.get('department')]
                else:
                    calc_df = self.employee_df
                
                # Custom risk calculation
                custom_risk = (
                    (6 - calc_df['job_satisfaction']) / 5 * job_sat_weight +
                    (6 - calc_df['performance_rating']) / 5 * perf_weight +
                    np.random.random(len(calc_df)) * dept_weight +  # Simplified dept risk
                    np.random.random(len(calc_df)) * pos_weight +   # Simplified position risk
                    np.random.random(len(calc_df)) * demo_weight    # Simplified demo risk
                )
                
                # Categorize with custom threshold
                high_risk_count = (custom_risk > threshold).sum()
                total_employees = len(calc_df)
                risk_percentage = (high_risk_count / total_employees) * 100
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Employees", total_employees)
                
                with col2:
                    st.metric("High Risk Count", high_risk_count)
                
                with col3:
                    st.metric("Risk Percentage", f"{risk_percentage:.1f}%")
                
                # Risk distribution chart
                fig = go.Figure(data=[go.Histogram(x=custom_risk, nbinsx=20)])
                fig.update_layout(
                    title="Custom Risk Score Distribution",
                    xaxis_title="Risk Score",
                    yaxis_title="Number of Employees",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                fig.add_vline(x=threshold, line_dash="dash", line_color="red", 
                             annotation_text="Risk Threshold")
                st.plotly_chart(fig, use_container_width=True)
            
            # Intervention strategies
            st.markdown("**🎯 Intervention Strategies by Risk Level:**")
            
            risk_strategies = {
                "Low Risk (0-30%)": {
                    "color": "#4facfe",
                    "strategies": [
                        "Recognition programs and peer appreciation",
                        "Career development planning sessions",
                        "Skill enhancement training opportunities",
                        "Regular check-ins and feedback sessions"
                    ]
                },
                "Medium Risk (30-60%)": {
                    "color": "#ffa726",
                    "strategies": [
                        "One-on-one mentoring programs",
                        "Flexible work arrangements",
                        "Project assignment optimization",
                        "Compensation and benefit reviews"
                    ]
                },
                "High Risk (60%+)": {
                    "color": "#ff6b6b",
                    "strategies": [
                        "Immediate manager intervention",
                        "Stay interviews and retention discussions",
                        "Role modification or transfer options",
                        "Urgent compensation adjustments"
                    ]
                }
            }
            
            for risk_level, data in risk_strategies.items():
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid {data['color']};">
                    <h4 style="color: {data['color']}; margin-bottom: 10px;">{risk_level}</h4>
                    <ul style="margin-left: 20px;">
                        {''.join([f'<li>{strategy}</li>' for strategy in data['strategies']])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    
    def show_employee_management_tab(self):
        """Employee management interface"""
        st.markdown('<h2 class="centered-title">👥 Employee Management</h2>', unsafe_allow_html=True)
        
        # Filter data based on user role
        if self.user_data.get('position') != 'Director':
            filtered_df = self.employee_df[self.employee_df['department'] == self.user_data.get('department')]
        else:
            filtered_df = self.employee_df
        
        # Search and filter
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_term = st.text_input("Search by Name", "")
        
        with col2:
            dept_filter = st.selectbox("Filter by Department", 
                                     ['All'] + list(filtered_df['department'].unique()))
        
        with col3:
            risk_filter = st.selectbox("Filter by Risk", 
                                     ['All'] + list(filtered_df['risk_category'].unique()))
        
        # Apply filters
        display_df = filtered_df.copy()
        
        if search_term:
            display_df = display_df[display_df['Username'].str.contains(search_term, case=False, na=False)]
        
        if dept_filter != 'All':
            display_df = display_df[display_df['department'] == dept_filter]
        
        if risk_filter != 'All':
            display_df = display_df[display_df['risk_category'] == risk_filter]
        
        # Display employee table
        st.markdown('<h3 class="section-header">Employee Directory</h3>', unsafe_allow_html=True)
        
        # Select columns to display
        display_columns = ['Employee_id', 'Username', 'department', 'position', 'age', 
                         'tenure', 'monthly_income', 'performance_rating', 'job_satisfaction', 
                         'risk_category']
        
        st.dataframe(display_df[display_columns], use_container_width=True)
        
        # Employee details
        if len(display_df) > 0:
            st.markdown('<h3 class="section-header">Employee Details</h3>', unsafe_allow_html=True)
            
            selected_employee = st.selectbox("Select Employee", display_df['Username'].tolist())
            
            if selected_employee:
                employee_data = display_df[display_df['Username'] == selected_employee].iloc[0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    **Employee ID:** {employee_data['Employee_id']}  
                    **Name:** {employee_data['Username']}  
                    **Department:** {employee_data['department']}  
                    **Position:** {employee_data['position']}  
                    **Age:** {employee_data['age']}  
                    **Tenure:** {employee_data['tenure']} years
                    """)
                
                with col2:
                    st.markdown(f"""
                    **Monthly Income:** ${employee_data['monthly_income']:,}  
                    **Performance Rating:** {employee_data['performance_rating']}/5  
                    **Job Satisfaction:** {employee_data['job_satisfaction']}/5  
                    **Risk Category:** {employee_data['risk_category']}
                    """)
    
    def show_reports_tab(self):
        """Reports and analytics tab"""
        st.markdown('<h2 class="centered-title">📋 Reports</h2>', unsafe_allow_html=True)
        
        # Filter data based on user role
        if self.user_data.get('position') != 'Director':
            filtered_df = self.employee_df[self.employee_df['department'] == self.user_data.get('department')]
        else:
            filtered_df = self.employee_df
        
        # Department summary
        st.markdown('<h3 class="section-header">Department Summary</h3>', unsafe_allow_html=True)
        
        dept_summary = filtered_df.groupby('department').agg({
            'Employee_id': 'count',
            'monthly_income': 'mean',
            'performance_rating': 'mean',
            'job_satisfaction': 'mean',
            'tenure': 'mean'
        }).round(2)
        
        dept_summary.columns = ['Employee Count', 'Avg Salary', 'Avg Performance', 'Avg Satisfaction', 'Avg Tenure']
        st.dataframe(dept_summary, use_container_width=True)
        
        # Risk analysis
        st.markdown('<h3 class="section-header">Risk Analysis</h3>', unsafe_allow_html=True)
        
        risk_analysis = filtered_df.groupby(['department', 'risk_category']).size().unstack(fill_value=0)
        
        fig = px.bar(risk_analysis, title="Risk Distribution by Department",
                    color_discrete_map={'Low': '#4facfe', 'Medium': '#ffa726', 'High': '#ff6b6b', 'Critical': '#d32f2f'})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Download reports
        st.markdown('<h3 class="section-header">Download Reports</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Download Employee Data"):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="employee_data.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("Download Department Summary"):
                csv = dept_summary.to_csv()
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="department_summary.csv",
                    mime="text/csv"
                )
