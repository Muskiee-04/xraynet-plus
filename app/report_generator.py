# report_generator.py
import io
import os
import sys
from datetime import datetime

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.cxr_recommendations import get_recommendation_detail
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import tempfile

class PDFReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.custom_styles = {}  # Use a separate dictionary for custom styles
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Setup custom styles for the PDF report without adding to main stylesheet"""
        # Title style
        self.custom_styles['Title'] = ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=colors.darkblue,
            spaceAfter=30,
            alignment=1  # Center aligned
        )
        
        # Heading style
        self.custom_styles['Heading'] = ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=12,
            textColor=colors.darkblue,
            spaceAfter=12
        )
        
        # Normal style
        self.custom_styles['Normal_Custom'] = ParagraphStyle(
            name='CustomNormal',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=12
        )
        
        # Bold style
        self.custom_styles['Bold'] = ParagraphStyle(
            name='CustomBold',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            textColor=colors.black
        )

    def generate_report(self, patient_data, analysis_results, output_path=None):
        """
        Generate a comprehensive PDF medical report
        
        Args:
            patient_data (dict): Patient information
            analysis_results (list): List of analysis results
            output_path (str): Optional output file path
        
        Returns:
            io.BytesIO: PDF file as bytes buffer
        """
        # Create buffer for PDF
        pdf_buffer = io.BytesIO()
        
        # Create PDF document
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Story holds all PDF elements
        story = []
        
        # Add header
        story.extend(self._create_header())
        
        # Add patient information
        story.extend(self._create_patient_section(patient_data))
        
        # Add analysis results
        story.extend(self._create_analysis_section(analysis_results))
        
        # Add summary and recommendations
        story.extend(self._create_summary_section(analysis_results))
        
        # Add footer
        story.extend(self._create_footer())
        
        # Build PDF
        doc.build(story)
        
        # Reset buffer position
        pdf_buffer.seek(0)
        
        # Save to file if output path provided
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(pdf_buffer.getvalue())
            pdf_buffer.seek(0)  # Reset after saving
        
        return pdf_buffer

    def _create_header(self):
        """Create report header"""
        elements = []
        
        # Title
        title = Paragraph("MEDICAL IMAGING ANALYSIS REPORT", self.custom_styles['Title'])
        elements.append(title)
        
        # Subtitle
        subtitle = Paragraph("XRAYNET+ AI Diagnostic System", self.styles['Heading2'])
        elements.append(subtitle)
        
        # Generation date
        date_str = datetime.now().strftime('%B %d, %Y at %H:%M:%S')
        date_para = Paragraph(f"Generated on: {date_str}", self.custom_styles['Normal_Custom'])
        elements.append(date_para)
        
        elements.append(Spacer(1, 20))
        
        return elements

    def _create_patient_section(self, patient_data):
        """Create patient information section"""
        elements = []
        
        # Section title
        elements.append(Paragraph("PATIENT INFORMATION", self.custom_styles['Heading']))
        elements.append(Spacer(1, 12))
        
        # Patient data table
        patient_info = [
            ["Patient ID:", str(patient_data.get("patient_id", "N/A"))],
            ["Name:", str(patient_data.get("name", "N/A"))],
            ["Age:", str(patient_data.get("age", "N/A"))],
            ["Gender:", str(patient_data.get("gender", "N/A"))],
            ["Clinical notes:", str(patient_data.get("clinical_notes", "N/A"))],
            ["Date of Birth:", str(patient_data.get("date_of_birth", "N/A"))],
            ["Referring Physician:", str(patient_data.get("physician", "N/A"))],
            ["Study Date:", str(patient_data.get("study_date", "N/A"))],
        ]
        
        patient_table = Table(patient_info, colWidths=[2*inch, 3*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(patient_table)
        elements.append(Spacer(1, 20))
        
        return elements

    def _create_analysis_section(self, analysis_results):
        """Create analysis results section"""
        elements = []
        
        elements.append(Paragraph("ANALYSIS RESULTS", self.custom_styles['Heading']))
        elements.append(Spacer(1, 12))
        
        if not analysis_results:
            elements.append(Paragraph("No analysis results available.", self.custom_styles['Normal_Custom']))
            return elements
        
        for i, result in enumerate(analysis_results, 1):
            # Result header
            elements.append(Paragraph(f"Image {i}: {result.get('filename', 'Unknown')}", 
                                    self.custom_styles['Bold']))
            
            pred = result.get('prediction', {})
            
            # Findings table
            findings_data = [
                ["Finding:", pred.get('class_name', 'No significant finding')],
                ["Confidence:", f"{pred.get('confidence', 0)*100:.1f}%"],
                ["Severity:", pred.get('severity', 'N/A')],
                ["Location:", pred.get('location', 'N/A')]
            ]
            
            findings_table = Table(findings_data, colWidths=[1.5*inch, 4*inch])
            findings_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.whitesmoke),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
            ]))
            
            elements.append(findings_table)
            
            # Description
            desc = pred.get('description', 'No description available.')
            elements.append(Paragraph(f"Description: {desc}", self.custom_styles['Normal_Custom']))
            
            elements.append(Spacer(1, 15))
        
        return elements

    def _create_summary_section(self, analysis_results):
        """Create summary and recommendations section"""
        elements = []
        
        elements.append(Paragraph("SUMMARY AND RECOMMENDATIONS", self.custom_styles['Heading']))
        elements.append(Spacer(1, 12))
        
        # Generate summary based on findings
        summary_text = self._generate_summary_text(analysis_results)
        elements.append(Paragraph(summary_text, self.custom_styles['Normal_Custom']))
        
        elements.append(Spacer(1, 12))
        
        elements.append(Paragraph("Recommendations (educational):", self.custom_styles['Bold']))
        elements.append(Spacer(1, 6))
        for block in self._recommendation_paragraphs(analysis_results):
            elements.append(block)
        
        elements.append(Spacer(1, 20))
        
        return elements

    def _create_footer(self):
        """Create report footer"""
        elements = []
        
        elements.append(Spacer(1, 30))
        
        # Disclaimer
        disclaimer = Paragraph(
            "This report was generated by XRAYNET+ AI system. "
            "This analysis should be reviewed and verified by a qualified healthcare professional. "
            "The findings are based on AI analysis and may require clinical correlation.",
            ParagraphStyle(
                name='Disclaimer',
                parent=self.styles['Normal'],
                fontSize=8,
                textColor=colors.grey,
                alignment=1
            )
        )
        
        elements.append(disclaimer)
        
        return elements

    def _generate_summary_text(self, analysis_results):
        """Generate summary text based on analysis results"""
        if not analysis_results:
            return "No significant findings detected in the analyzed images."
        
        # Count different types of findings
        findings = []
        for result in analysis_results:
            pred = result.get('prediction', {})
            finding = pred.get('class_name', '').lower()
            benign = ('normal', 'no finding', 'healthy', 'no findings')
            if finding and not any(b in finding for b in benign) and finding not in findings:
                findings.append(finding)
        
        if not findings:
            return "No significant pathological findings detected. Images appear within normal limits."
        
        if len(findings) == 1:
            return f"Analysis revealed {findings[0]}. Further evaluation recommended."
        else:
            return f"Analysis revealed multiple findings including {', '.join(findings[:-1])} and {findings[-1]}."

    def _recommendation_paragraphs(self, analysis_results):
        """Build ReportLab flowables for class-specific clinical and prevention guidance."""
        out: list = []
        if not analysis_results:
            out.append(Paragraph("No specific recommendations based on current analysis.", self.custom_styles['Normal_Custom']))
            return out

        seen: set[str] = set()
        for result in analysis_results:
            pred = result.get("prediction", {})
            name = (pred.get("class_name") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)

            detail = get_recommendation_detail(name)
            out.append(Paragraph(f"<b>{name}</b>", self.custom_styles['Bold']))
            out.append(Paragraph(detail["summary"], self.custom_styles['Normal_Custom']))
            out.append(Spacer(1, 6))
            out.append(Paragraph("<b>Clinical next steps</b>", self.custom_styles['Bold']))
            for line in detail["clinical_steps"]:
                out.append(Paragraph(f"• {line}", self.custom_styles['Normal_Custom']))
            out.append(Spacer(1, 6))
            out.append(Paragraph("<b>Prevention &amp; wellness</b>", self.custom_styles['Bold']))
            for line in detail["prevention"]:
                out.append(Paragraph(f"• {line}", self.custom_styles['Normal_Custom']))
            out.append(Spacer(1, 12))

        out.append(Paragraph(
            "<i>Correlation with examination and laboratory testing is essential. "
            "This text is educational and not a substitute for a licensed clinician.</i>",
            self.custom_styles['Normal_Custom'],
        ))
        return out

    def generate_simple_report(self, patient_data, results):
        """
        Generate a simple text-based report (fallback method)
        
        Args:
            patient_data (dict): Patient information
            results (list): Analysis results
        
        Returns:
            io.BytesIO: Simple text report as bytes buffer
        """
        report_content = f"""
MEDICAL REPORT - XRAYNET+
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PATIENT INFORMATION:
- Patient ID: {patient_data.get('patient_id', 'N/A')}
- Age: {patient_data.get('age', 'N/A')}
- Gender: {patient_data.get('gender', 'N/A')}

FINDINGS:
"""
        
        for i, result in enumerate(results, 1):
            pred = result.get('prediction', {})
            detail = get_recommendation_detail(pred.get('class_name', ''))
            clin = "\n".join(f"  - {s}" for s in detail["clinical_steps"])
            prev = "\n".join(f"  - {s}" for s in detail["prevention"])
            report_content += f"""
Image {i}: {result.get('filename', 'Unknown')}
- Primary Finding: {pred.get('class_name', 'Unknown')}
- Confidence: {pred.get('confidence', 0)*100:.1f}%
- Description: {pred.get('description', 'N/A')}
- Summary: {pred.get('recommendation', 'N/A')}
- Clinical next steps:
{clin}
- Prevention & wellness:
{prev}
"""
        
        # Convert to bytes
        pdf_buffer = io.BytesIO()
        pdf_buffer.write(report_content.encode('utf-8'))
        pdf_buffer.seek(0)
        
        return pdf_buffer


# For backward compatibility
def generate_report(patient_data, results, output_path=None):
    """
    Standalone function for backward compatibility
    """
    generator = PDFReportGenerator()
    return generator.generate_report(patient_data, results, output_path)