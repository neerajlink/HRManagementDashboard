import streamlit as st
import pandas as pd
import hashlib

class AuthManager:
    def __init__(self):
        pass
    
    def authenticate(self, username, password, employee_df):
        """
        Authenticate user against employee database
        """
        try:
            # Find user in employee database
            user_record = employee_df[employee_df['Username'] == username]
            
            if not user_record.empty:
                # Check password (stored as string in CSV)
                stored_password = str(user_record.iloc[0]['Password'])
                
                if password == stored_password:
                    # Return user data
                    return user_record.iloc[0].to_dict()
            
            return None
            
        except Exception as e:
            st.error(f"Authentication error: {str(e)}")
            return None
    
    def logout(self):
        """
        Clear session state for logout
        """
        st.session_state.authenticated = False
        st.session_state.user_data = None
        st.success("Logged out successfully!")
    
    def is_admin(self, user_data):
        """
        Check if user has admin privileges (Directors and Managers)
        """
        if user_data and 'position' in user_data:
            return user_data['position'] in ['Director', 'Manager']
        return False
    
    def get_user_permissions(self, user_data):
        """
        Get user permissions based on role
        """
        if not user_data:
            return {'view_dashboard': False, 'view_analytics': False, 'manage_employees': False}
        
        position = user_data.get('position', '')
        
        if position in ['Director']:
            return {
                'view_dashboard': True,
                'view_analytics': True,
                'manage_employees': True,
                'view_reports': True,
                'ml_predictions': True
            }
        elif position in ['Manager']:
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
