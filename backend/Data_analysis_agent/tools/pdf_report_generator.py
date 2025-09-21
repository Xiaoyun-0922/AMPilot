"""
PDF Report Generator for Data Analysis Agent

This module generates professional PDF reports from analysis results
using ReportLab with custom styling and layout.
"""

import os
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether
)
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

logger = logging.getLogger(__name__)

class PDFReportGenerator:
    """Professional PDF report generator for data analysis results"""
    
    def __init__(self, results_dir: str):
        """Initialize the PDF report generator"""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        logger.info("PDF Report Generator initialized")
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#2E86AB'),
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=20,
            spaceBefore=20,
            textColor=colors.HexColor('#A23B72'),
            fontName='Helvetica-Bold'
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=16,
            textColor=colors.HexColor('#F18F01'),
            fontName='Helvetica-Bold'
        ))
        
        # Body text with better spacing
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            alignment=TA_JUSTIFY,
            fontName='Helvetica'
        ))
        
        # Code style
        self.styles.add(ParagraphStyle(
            name='CodeStyle',
            parent=self.styles['Code'],
            fontSize=9,
            spaceAfter=8,
            spaceBefore=8,
            backColor=colors.HexColor('#F5F5F5'),
            borderColor=colors.HexColor('#CCCCCC'),
            borderWidth=1,
            borderPadding=8,
            fontName='Courier'
        ))
        
        # Highlight style
        self.styles.add(ParagraphStyle(
            name='Highlight',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            backColor=colors.HexColor('#E8F4FD'),
            borderColor=colors.HexColor('#2E86AB'),
            borderWidth=1,
            borderPadding=8,
            fontName='Helvetica'
        ))
    
    def generate_report(self, analysis_result: Dict[str, Any], session_id: str = None) -> str:
        """
        Generate a comprehensive PDF report from analysis results
        
        Args:
            analysis_result: Complete analysis result dictionary
            session_id: Optional session identifier
            
        Returns:
            Path to the generated PDF file
        """
        try:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_suffix = f"_{session_id}" if session_id else ""
            filename = f"data_analysis_report_{timestamp}{session_suffix}.pdf"
            filepath = self.results_dir / filename
            
            # Create PDF document
            doc = SimpleDocTemplate(
                str(filepath),
                pagesize=A4,
                rightMargin=2*cm,
                leftMargin=2*cm,
                topMargin=2*cm,
                bottomMargin=2*cm
            )
            
            # Build content
            story = []

            # Header
            self._add_header(story, analysis_result)

            # Executive Summary
            self._add_executive_summary(story, analysis_result)

            # Main Results
            self._add_main_results(story, analysis_result)

            # Key Findings
            self._add_key_findings(story, analysis_result)

            # Recommendations
            self._add_recommendations(story, analysis_result)

            # Technical Details (simplified)
            self._add_technical_details(story, analysis_result)

            # Footer
            self._add_footer(story)
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"PDF report generated: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}")
            raise
    
    def _add_header(self, story: List, analysis_result: Dict[str, Any]):
        """Add report header"""
        # Title
        story.append(Paragraph("📊 Data Analysis Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.3*inch))
        
        # Metadata table
        timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        success_status = "✅ Completed Successfully" if analysis_result.get('success', False) else "❌ Completed with Issues"
        
        metadata = [
            ['Report Generated:', timestamp],
            ['Analysis Status:', success_status],
            ['Current Step:', analysis_result.get('current_step', 'Unknown')],
        ]
        
        table = Table(metadata, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F0F0F0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#CCCCCC')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        
        story.append(table)
        story.append(Spacer(1, 0.3*inch))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#2E86AB')))
        story.append(Spacer(1, 0.2*inch))
    
    def _add_executive_summary(self, story: List, analysis_result: Dict[str, Any]):
        """Add executive summary section"""
        story.append(Paragraph("📋 Executive Summary", self.styles['CustomSubtitle']))

        # Extract key insights from final analysis
        final_analysis = analysis_result.get('final_analysis', '')
        if final_analysis:
            # Create a condensed summary with key findings
            summary_text = self._create_executive_summary(final_analysis)
            story.append(Paragraph(summary_text, self.styles['Highlight']))
        else:
            story.append(Paragraph("Analysis completed successfully. Detailed results are provided in the following sections.", self.styles['CustomBody']))

        story.append(Spacer(1, 0.3*inch))
    
    def _add_main_results(self, story: List, analysis_result: Dict[str, Any]):
        """Add main results section with clean formatting"""
        final_analysis = analysis_result.get('final_analysis', '')
        if final_analysis:
            story.append(Paragraph("📊 Analysis Results", self.styles['CustomSubtitle']))

            # Clean and format the analysis text
            clean_text = self._format_analysis_text(final_analysis)
            story.append(Paragraph(clean_text, self.styles['CustomBody']))
            story.append(Spacer(1, 0.3*inch))
    
    def _add_key_findings(self, story: List, analysis_result: Dict[str, Any]):
        """Add key findings section"""
        final_analysis = analysis_result.get('final_analysis', '')
        if final_analysis:
            story.append(Paragraph("🔍 Key Findings", self.styles['CustomSubtitle']))

            # Extract key findings from analysis
            key_findings = self._extract_key_findings(final_analysis)
            for finding in key_findings:
                story.append(Paragraph(f"• {finding}", self.styles['CustomBody']))

            story.append(Spacer(1, 0.3*inch))
    
    def _add_technical_details(self, story: List, analysis_result: Dict[str, Any]):
        """Add technical details section (simplified)"""
        story.append(Paragraph("🔧 Technical Details", self.styles['CustomSubtitle']))

        # Add task understanding if available
        task_understanding = analysis_result.get('task_understanding', '')
        if task_understanding:
            story.append(Paragraph("Analysis Objective:", self.styles['SectionHeader']))
            text_content = self._extract_text_content(task_understanding)
            clean_text = self._clean_text(text_content)[:300] + "..." if len(self._clean_text(text_content)) > 300 else self._clean_text(text_content)
            story.append(Paragraph(clean_text, self.styles['CustomBody']))
            story.append(Spacer(1, 0.1*inch))

        # Add execution status
        execution_result = analysis_result.get('execution_result', {})
        if execution_result:
            if execution_result.get('success'):
                story.append(Paragraph("✅ Analysis executed successfully", self.styles['CustomBody']))
            else:
                story.append(Paragraph("❌ Analysis encountered issues", self.styles['CustomBody']))

        story.append(Spacer(1, 0.2*inch))
    

    
    def _add_recommendations(self, story: List, analysis_result: Dict[str, Any]):
        """Add recommendations section"""
        recommendations = analysis_result.get('recommendations', [])
        if recommendations:
            story.append(Paragraph("💡 Recommendations", self.styles['CustomSubtitle']))
            
            for i, rec in enumerate(recommendations, 1):
                story.append(Paragraph(f"{i}. {rec}", self.styles['CustomBody']))
            
            story.append(Spacer(1, 0.2*inch))
    
    def _add_footer(self, story: List):
        """Add report footer"""
        story.append(Spacer(1, 0.3*inch))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CCCCCC')))
        story.append(Spacer(1, 0.1*inch))
        
        footer_text = "Generated by AMPilot Data Analysis Agent | Powered by AI-driven insights"
        story.append(Paragraph(footer_text, self.styles['Normal']))
    
    def _create_executive_summary(self, text: str) -> str:
        """Create a concise executive summary"""
        # Remove markdown formatting and extract key points
        clean_text = self._clean_markdown(text)

        # Extract first paragraph or key findings
        paragraphs = clean_text.split('\n\n')
        summary_parts = []

        for para in paragraphs[:2]:  # Take first 2 paragraphs
            if para.strip() and len(para.strip()) > 20:
                # Limit length
                if len(para) > 200:
                    para = para[:200] + "..."
                summary_parts.append(para.strip())

        return ' '.join(summary_parts) or "Analysis completed successfully with detailed findings."

    def _format_analysis_text(self, text: str) -> str:
        """Format analysis text for better PDF presentation"""
        # Clean markdown and format for PDF
        clean_text = self._clean_markdown(text)

        # Split into paragraphs and clean each
        paragraphs = clean_text.split('\n\n')
        formatted_paragraphs = []

        for para in paragraphs:
            if para.strip():
                # Remove excessive whitespace
                para = ' '.join(para.split())
                formatted_paragraphs.append(para)

        return '\n\n'.join(formatted_paragraphs)

    def _extract_key_findings(self, text: str) -> List[str]:
        """Extract key findings as bullet points"""
        findings = []
        clean_text = self._clean_markdown(text)

        # Look for numbered points, bullet points, or key phrases
        lines = clean_text.split('\n')

        for line in lines:
            line = line.strip()
            if line and (
                line.startswith(('1.', '2.', '3.', '4.', '5.')) or
                line.startswith(('•', '-', '*')) or
                'correlation' in line.lower() or
                'significant' in line.lower() or
                'finding' in line.lower()
            ):
                # Clean and limit length
                clean_line = line.lstrip('123456789.-•* ').strip()
                if len(clean_line) > 100:
                    clean_line = clean_line[:100] + "..."
                if clean_line:
                    findings.append(clean_line)

        # If no specific findings found, extract first few sentences
        if not findings:
            sentences = clean_text.split('.')[:3]
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(sentence) > 20:
                    findings.append(sentence)

        return findings[:5]  # Limit to 5 findings

    def _clean_markdown(self, text: str) -> str:
        """Clean markdown formatting from text"""
        if not text:
            return ""

        # Remove markdown headers
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

        # Remove bold/italic
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)

        # Remove code blocks
        text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
        text = re.sub(r'`([^`]+)`', r'\1', text)

        # Clean up whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)

        return text.strip()
    
    def _clean_text(self, text: str) -> str:
        """Clean text for PDF rendering"""
        if not text:
            return ""
        
        # Remove markdown formatting
        text = text.replace('**', '').replace('*', '')
        text = text.replace('###', '').replace('##', '').replace('#', '')
        
        # Handle special characters
        text = self._escape_html(text)
        
        return text.strip()

    def _extract_text_content(self, content) -> str:
        """Extract text content from various data formats"""
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            # Try to extract meaningful text from dict
            if 'raw_response' in content:
                return content['raw_response']
            elif 'content' in content:
                return content['content']
            elif 'text' in content:
                return content['text']
            else:
                # Convert dict to readable format
                return json.dumps(content, indent=2)
        elif isinstance(content, list):
            return '\n'.join(str(item) for item in content)
        else:
            return str(content)

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters"""
        if not text:
            return ""
        
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        text = text.replace('"', '&quot;')
        text = text.replace("'", '&#x27;')
        
        return text
