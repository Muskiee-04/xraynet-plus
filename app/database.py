# app/database.py
from __future__ import annotations

import sqlite3
import json
import os
from datetime import datetime
import numpy as np
import pandas as pd

class DatabaseManager:
    def __init__(self, db_path: str | None = None):
        if db_path is None:
            os.makedirs(os.path.join(os.path.dirname(__file__), "..", "data"), exist_ok=True)
            db_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "data", "xraynet_plus.db")
            )
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Patients table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                name TEXT,
                age INTEGER,
                gender TEXT,
                clinical_notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Examinations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS examinations (
                examination_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                timestamp TEXT,
                total_images INTEGER,
                primary_finding TEXT,
                average_confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
            )
        ''')
        
        # Image analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_analysis (
                analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
                examination_id INTEGER,
                filename TEXT,
                prediction_data TEXT,  -- JSON string with all prediction data
                heatmap_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (examination_id) REFERENCES examinations (examination_id)
            )
        ''')
        
        # Reports table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reports (
                report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                examination_id INTEGER,
                report_path TEXT,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (examination_id) REFERENCES examinations (examination_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_patient_data(self, patient_data):
        """Save or update patient information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO patients 
                (patient_id, name, age, gender, clinical_notes, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                patient_data.get('patient_id'),
                patient_data.get('name', ''),
                patient_data.get('age', 0),
                patient_data.get('gender', ''),
                patient_data.get('clinical_notes', ''),
                datetime.now()
            ))
            
            conn.commit()
            return True
        except Exception as e:
            print(f"Error saving patient data: {e}")
            return False
        finally:
            conn.close()
    
    def store_examination(self, patient_data, results):
        """Store complete examination data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # First, save patient data
            self.save_patient_data(patient_data)
            
            # Calculate examination summary
            total_images = len(results)
            primary_findings = [r['prediction']['class_name'] for r in results]
            primary_finding = max(set(primary_findings), key=primary_findings.count)
            avg_confidence = sum([r['prediction']['confidence'] for r in results]) / total_images
            
            # Store examination record
            cursor.execute('''
                INSERT INTO examinations 
                (patient_id, timestamp, total_images, primary_finding, average_confidence)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                patient_data.get('patient_id'),
                patient_data.get('timestamp', datetime.now().isoformat()),
                total_images,
                primary_finding,
                avg_confidence
            ))
            
            examination_id = cursor.lastrowid
            
            # Store individual image analyses
            for result in results:
                prediction_json = json.dumps(result['prediction'], default=str)
                
                # Save heatmap image if available
                heatmap_path = None
                if 'heatmap' in result:
                    heatmap_filename = f"heatmap_{examination_id}_{result['filename']}.png"
                    heatmap_path = self._save_heatmap_image(result['heatmap'], heatmap_filename)
                
                cursor.execute('''
                    INSERT INTO image_analysis 
                    (examination_id, filename, prediction_data, heatmap_path)
                    VALUES (?, ?, ?, ?)
                ''', (
                    examination_id,
                    result['filename'],
                    prediction_json,
                    heatmap_path
                ))
            
            conn.commit()
            return examination_id
        except Exception as e:
            print(f"Error storing examination: {e}")
            return None
        finally:
            conn.close()
    
    def _save_heatmap_image(self, heatmap_array, filename):
        """Save heatmap image to file system"""
        try:
            # Create heatmaps directory if it doesn't exist
            heatmaps_dir = "heatmaps"
            if not os.path.exists(heatmaps_dir):
                os.makedirs(heatmaps_dir)
            
            filepath = os.path.join(heatmaps_dir, filename)
            
            # Convert numpy array to PIL Image and save
            from PIL import Image
            if heatmap_array.dtype != np.uint8:
                heatmap_array = (heatmap_array * 255).astype(np.uint8)
            
            heatmap_image = Image.fromarray(heatmap_array)
            heatmap_image.save(filepath)
            
            return filepath
        except Exception as e:
            print(f"Error saving heatmap: {e}")
            return None
    
    def get_patient_data(self, patient_id):
        """Retrieve patient information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT * FROM patients WHERE patient_id = ?', (patient_id,))
            patient = cursor.fetchone()
            
            if patient:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, patient))
            return None
        finally:
            conn.close()
    
    def get_patient_examinations(self, patient_id):
        """Get all examinations for a patient"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT * FROM examinations 
                WHERE patient_id = ? 
                ORDER BY created_at DESC
            ''', (patient_id,))
            
            examinations = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            return [dict(zip(columns, exam)) for exam in examinations]
        finally:
            conn.close()
    
    def get_examination_details(self, examination_id):
        """Get detailed analysis for a specific examination"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get examination info
            cursor.execute('SELECT * FROM examinations WHERE examination_id = ?', (examination_id,))
            exam = cursor.fetchone()
            
            if not exam:
                return None
            
            columns = [desc[0] for desc in cursor.description]
            examination_data = dict(zip(columns, exam))
            
            # Get image analyses
            cursor.execute('''
                SELECT * FROM image_analysis 
                WHERE examination_id = ? 
                ORDER BY analysis_id
            ''', (examination_id,))
            
            analyses = cursor.fetchall()
            analysis_columns = [desc[0] for desc in cursor.description]
            
            examination_data['analyses'] = []
            for analysis in analyses:
                analysis_dict = dict(zip(analysis_columns, analysis))
                # Parse JSON prediction data
                if analysis_dict['prediction_data']:
                    analysis_dict['prediction_data'] = json.loads(analysis_dict['prediction_data'])
                examination_data['analyses'].append(analysis_dict)
            
            return examination_data
        finally:
            conn.close()
    
    def get_all_patients(self):
        """Get list of all patients"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT p.*, 
                       COUNT(e.examination_id) as total_examinations,
                       MAX(e.created_at) as last_examination
                FROM patients p
                LEFT JOIN examinations e ON p.patient_id = e.patient_id
                GROUP BY p.patient_id
                ORDER BY p.created_at DESC
            ''')
            
            patients = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            return [dict(zip(columns, patient)) for patient in patients]
        finally:
            conn.close()
    
    def get_statistics(self):
        """Get overall system statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            stats = {}
            
            # Total patients
            cursor.execute('SELECT COUNT(*) FROM patients')
            stats['total_patients'] = cursor.fetchone()[0]
            
            # Total examinations
            cursor.execute('SELECT COUNT(*) FROM examinations')
            stats['total_examinations'] = cursor.fetchone()[0]
            
            # Total images analyzed
            cursor.execute('SELECT COUNT(*) FROM image_analysis')
            stats['total_images'] = cursor.fetchone()[0]
            
            # Most common findings
            cursor.execute('''
                SELECT primary_finding, COUNT(*) as count 
                FROM examinations 
                GROUP BY primary_finding 
                ORDER BY count DESC
            ''')
            stats['common_findings'] = cursor.fetchall()
            
            return stats
        finally:
            conn.close()
    
    def export_to_csv(self, table_name, filename=None):
        """Export table data to CSV"""
        if filename is None:
            filename = f"{table_name}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            df.to_csv(filename, index=False)
            return filename
        finally:
            conn.close()
    
    def backup_database(self, backup_path=None):
        """Create a backup of the database"""
        if backup_path is None:
            backup_path = f"xraynet_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        
        try:
            conn = sqlite3.connect(self.db_path)
            backup_conn = sqlite3.connect(backup_path)
            
            conn.backup(backup_conn)
            
            conn.close()
            backup_conn.close()
            
            return backup_path
        except Exception as e:
            print(f"Error creating backup: {e}")
            return None
