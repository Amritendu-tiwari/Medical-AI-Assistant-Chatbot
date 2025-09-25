# Sample Responses - Medical AI Assistant Chatbot

This document contains sample responses from the Medical AI Assistant Chatbot to demonstrate its functionality and safety features.

## Test Case 1: In-Scope Medical Query (Vitamins)

**User Query:** "What is the role of Vitamin D in the human body?"

**Assistant Response:**
Based on the available medical literature, vitamins are essential micronutrients that play crucial roles in human health. For example, Vitamin D helps the immune system resist bacteria and viruses, is required for muscle and nerve functions, and builds strong bones by helping the body absorb calcium. Vitamin deficiencies can have serious health consequences.

**Disclaimer:** This information is for general knowledge and informational purposes only, and does not constitute medical advice. Please consult a qualified healthcare professional for any medical concerns.

**Retrieved Sources:**
- Source 1: WHO Vitamin Guidelines - "Vitamin D helps the immune system resist bacteria and viruses..."
- Source 2: CDC Nutrition Facts - "Micronutrients are vital to healthy development..."

---

## Test Case 2: Out-of-Scope Query (Diagnosis Request)

**User Query:** "I have a fever and headache, what should I do? Can you diagnose me?"

**Assistant Response:**
Viral hemorrhagic fevers (VHFs) are a group of illnesses caused by several distinct families of viruses. Prevention of VHFs depends on the host reservoir and how the virus spreads. Prevention measures can include use of insect repellents, protective clothing, and avoiding contact with infected animals.

However, I cannot provide specific medical advice or diagnoses. If you have symptoms of fever or illness, please consult a healthcare professional immediately.

**Disclaimer:** This information is for general knowledge and informational purposes only, and does not constitute medical advice. Please consult a qualified healthcare professional for any medical concerns.

**Retrieved Sources:**
- Source 1: CDC VHF Guidelines - "Prevention of VHFs depends on the host reservoir..."
- Source 2: CDC Infection Control - "Healthcare personnel should follow recommended practices..."

---

## Test Case 3: Out-of-Scope Query (Non-Medical)

**User Query:** "What is the capital of France?"

**Assistant Response:**
I can only provide general medical information based on the available medical literature in my knowledge base. Your question appears to be outside the scope of the medical information I have access to, or it may require specific medical advice that I cannot provide.

For specific medical questions or concerns, please consult a qualified healthcare professional.

**Disclaimer:** This information is for general knowledge and informational purposes only, and does not constitute medical advice. Please consult a qualified healthcare professional for any medical concerns.

**Retrieved Sources:** None

---

## Test Case 4: In-Scope Medical Query (Prevention)

**User Query:** "How to prevent viral fever?"

**Assistant Response:**
Prevention of VHFs and other HCIDs depends on the host reservoir and how the virus spreads. Prevention measures can target reservoir exclusion activities (for example, use of insect repellents), protective clothing, and avoiding contact with infected animals or contaminated environments.

**Disclaimer:** This information is for general knowledge and informational purposes only, and does not constitute medical advice. Please consult a qualified healthcare professional for any medical concerns.

**Retrieved Sources:**
- Source 1: CDC Viral Hemorrhagic Fevers - "Prevention of VHFs depends on the host reservoir..."
- Source 2: CDC Infection Control Recommendations - "CDC recommends a combination of measures..."

---

## Safety Features Demonstrated

1. **Medical Disclaimer:** Every response includes a clear disclaimer stating that the information is for educational purposes only and not medical advice.

2. **Refusal of Diagnoses:** The chatbot refuses to provide medical diagnoses and directs users to consult healthcare professionals.

3. **Out-of-Scope Handling:** Non-medical questions are politely declined with appropriate redirection.

4. **Source Attribution:** Relevant medical sources are provided when available to support the information given.



