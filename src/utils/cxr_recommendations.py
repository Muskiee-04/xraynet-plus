"""
Condition-specific educational guidance for XRAYNET+ UI and reports.

Not a substitute for professional medical advice; always correlate with a clinician.
"""
from __future__ import annotations

def _match_class(name: str) -> str:
    """Map model label to canonical key."""
    n = name.strip().lower()
    if "tuberculosis" in n or n == "tb":
        return "Tuberculosis"
    if "pneumonia" in n:
        return "Pneumonia"
    if "covid" in n:
        return "COVID-19"
    if "no finding" in n:
        return "No Findings"
    return "No Findings"


def get_recommendation_detail(class_name: str) -> dict[str, str | list[str]]:
    """
    Returns summary line plus bullet lists for clinical follow-up and prevention/wellness.
    """
    key = _match_class(class_name)

    if key == "Tuberculosis":
        return {
            "summary": (
                "Urgent clinical evaluation is appropriate: TB must be confirmed or ruled out "
                "with appropriate testing and public-health follow-up per local protocol."
            ),
            "clinical_steps": [
                "See a physician promptly; active TB is a public-health concern and needs "
                "confirmed diagnosis (e.g., microbiology / molecular tests as available).",
                "Follow local infection-control guidance (e.g., masking, ventilation, "
                "avoiding close contact) until a clinician advises otherwise.",
                "Discuss HIV and other immunocompromise risk factors with your care team "
                "when relevant.",
                "If TB treatment is started, complete the full course exactly as prescribed "
                "to prevent resistance and relapse.",
            ],
            "prevention": [
                "Where recommended, ensure BCG vaccination per national policy (usually in childhood).",
                "Avoid prolonged close contact with people who have untreated, suspected infectious TB.",
                "Improve indoor ventilation where possible; good nutrition supports immune health.",
                "If you have latent TB, discuss preventive therapy with your clinician.",
            ],
        }

    if key == "Pneumonia":
        return {
            "summary": (
                "Pneumonia on imaging warrants clinical correlation; severity and cause "
                "(bacterial vs viral vs other) guide treatment and follow-up."
            ),
            "clinical_steps": [
                "Seek timely medical assessment—vitals, oxygen saturation, and examination matter.",
                "Antibiotics are used when bacterial pneumonia is suspected or confirmed; "
                "follow your prescriber’s plan and duration.",
                "Return or seek urgent care for worsening breathlessness, confusion, "
                "persistent high fever, or inability to keep fluids down.",
                "A follow-up chest X-ray may be advised if symptoms linger or if you have risk factors.",
            ],
            "prevention": [
                "Stay up to date with influenza and pneumococcal vaccines per age and health status.",
                "Wash hands often; avoid smoking and second-hand smoke.",
                "Manage chronic conditions (COPD, diabetes, heart disease) that raise pneumonia risk.",
                "Good sleep, nutrition, and exercise support lung health.",
            ],
        }

    if key == "COVID-19":
        return {
            "summary": (
                "If COVID-19 is suspected or modeled as likely, follow current public-health "
                "and clinical guidance for testing, isolation, and treatment eligibility."
            ),
            "clinical_steps": [
                "Use local/national guidance for testing and reporting; isolate if instructed "
                "to reduce spread to vulnerable people.",
                "Monitor breathing effort, oxygen level if available, hydration, and alertness; "
                "seek emergency care for severe shortness of breath, chest pain, confusion, or bluish lips.",
                "Ask a clinician whether antiviral or other therapies are appropriate for your age "
                "and risk profile.",
                "Stay in touch with your doctor if symptoms worsen or persist beyond expectations.",
            ],
            "prevention": [
                "Stay current with recommended COVID-19 boosters for your age and health status.",
                "Improve ventilation in shared spaces; consider masks in crowded indoor settings "
                "during surges or if you are high-risk.",
                "Stay home when ill; cover coughs/sneezes and wash hands frequently.",
                "Protect higher-risk household members by reducing exposure when you have respiratory symptoms.",
            ],
        }

    # No Findings
    return {
        "summary": (
            "The model favors no acute finding on this image; symptoms and examination "
            "still determine next steps."
        ),
        "clinical_steps": [
            "Correlate with symptoms, examination, and labs; imaging can be normal early or in some conditions.",
            "If symptoms persist (cough, fever, weight loss, night sweats), return for reassessment "
            "even if the X-ray appears clear.",
            "Maintain routine follow-up for chronic lung or heart conditions as your clinician advises.",
        ],
        "prevention": [
            "Do not smoke; avoid vaping and second-hand smoke; consider flu and pneumococcal vaccines as advised.",
            "Exercise regularly; manage weight, blood pressure, and diabetes to support heart and lung health.",
            "Practice hand hygiene; seek care for prolonged respiratory symptoms.",
        ],
    }


def get_recommendation_line(class_name: str) -> str:
    """Single-line summary for API/backward compatibility."""
    return str(get_recommendation_detail(class_name)["summary"])
