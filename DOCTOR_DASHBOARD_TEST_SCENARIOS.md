# Doctor Dashboard Test Scenarios Documentation

## Overview
This document provides comprehensive test scenarios for the Doctor Dashboard based on the UHC (UnitedHealthcare) rules engine. The scenarios cover various medical service requests, patient profiles, and expected system responses.

## File Structure
- `sample_inputs_doctor_dashboard.json` - Contains sample input requests for testing
- `sample_outputs_doctor_dashboard.json` - Contains expected system responses
- `DOCTOR_DASHBOARD_TEST_SCENARIOS.md` - This documentation file

## Test Scenarios

### Scenario 001: Approved Statin Medication - Low Risk Patient
**Description**: A low-risk patient requesting a tier 1 statin medication
- **Patient**: 45-year-old male, low risk score (0.3)
- **Service**: Atorvastatin (Tier 1, no PA required, low cost)
- **Expected Result**: **APPROVED** - "Low-risk, low-tier service"
- **Appeal Prediction**: No appeal recommended

### Scenario 002: GLP-1 Medication with Step Therapy Required
**Description**: A patient requesting Semaglutide which requires step therapy
- **Patient**: 52-year-old female, medium risk score (0.6)
- **Service**: Semaglutide (Tier 3, requires PA, high cost, step therapy)
- **Expected Result**: **NEEDS DOCS** - "Step therapy required before approval"
- **Appeal Prediction**: No appeal recommended

### Scenario 003: High-Cost PCSK9 Inhibitor - High Risk Patient
**Description**: A high-risk patient requesting expensive PCSK9 inhibitor
- **Patient**: 68-year-old male, high risk score (0.8)
- **Service**: Evolocumab (Tier 4, requires PA, high cost)
- **Expected Result**: **NEEDS DOCS** - "High-cost service for high-risk member. Need docs."
- **Appeal Prediction**: No appeal recommended

### Scenario 004: Biologic Medication - Rheumatoid Arthritis
**Description**: A patient requesting biologic medication for RA
- **Patient**: 41-year-old female, medium risk score (0.5)
- **Service**: Adalimumab (Tier 5, requires PA, high cost, step therapy)
- **Expected Result**: **NEEDS DOCS** - "Step therapy required before approval"
- **Appeal Prediction**: No appeal recommended

### Scenario 005: Oncology Medication - High Tier
**Description**: A high-risk patient requesting oncology medication
- **Patient**: 55-year-old male, very high risk score (0.9)
- **Service**: Imatinib (Tier 6, requires PA, high cost)
- **Expected Result**: **NEEDS DOCS** - "High tier service requires additional review"
- **Appeal Prediction**: No appeal recommended

### Scenario 006: Emergency Service - Auto-Approved
**Description**: An emergency cardiac procedure
- **Patient**: 29-year-old female, low risk score (0.2)
- **Service**: Emergency Cardiac Catheterization (Emergency type)
- **Expected Result**: **APPROVED** - "Emergency/urgent service auto-approved"
- **Appeal Prediction**: No appeal recommended

### Scenario 007: Denied Medication - Not Covered
**Description**: A medication that doesn't meet coverage criteria
- **Patient**: 35-year-old male, medium risk score (0.4)
- **Service**: Tirzepatide (Tier 3, requires PA, high cost, step therapy)
- **Expected Result**: **DENIED** - "Service not medically necessary under plan criteria"
- **Appeal Prediction**: **HIGH RISK** - Appeal recommended with 85% confidence

### Scenario 008: Imaging Service - Needs Documentation
**Description**: An MRI scan requiring additional documentation
- **Patient**: 48-year-old female, medium risk score (0.6)
- **Service**: MRI Brain with Contrast (Tier 2, requires PA, high cost)
- **Expected Result**: **NEEDS DOCS** - "Additional documentation required"
- **Appeal Prediction**: No appeal recommended

### Scenario 009: DME Service - Low Risk Patient
**Description**: A low-risk patient requesting DME
- **Patient**: 72-year-old male, low risk score (0.3)
- **Service**: Standard Wheelchair (Tier 1, no PA required, low cost)
- **Expected Result**: **APPROVED** - "Low-risk, low-tier service"
- **Appeal Prediction**: No appeal recommended

### Scenario 010: Procedure Service - High Tier
**Description**: A high-tier surgical procedure
- **Patient**: 38-year-old female, high risk score (0.7)
- **Service**: Arthroscopic Knee Surgery (Tier 4, requires PA, high cost, step therapy)
- **Expected Result**: **NEEDS DOCS** - "High tier service requires additional review"
- **Appeal Prediction**: No appeal recommended

## Additional Test Scenarios

### Scenario 011: Florida State Policy - Requires PA Documentation
**Description**: A medication request in Florida requiring state-specific PA documentation
- **Patient**: 44-year-old male, medium risk score (0.5), Florida resident
- **Service**: Celecoxib (Tier 2, requires PA)
- **Expected Result**: **NEEDS DOCS** - "State policy requires PA documentation"
- **Appeal Prediction**: No appeal recommended

### Scenario 012: Very High Tier Service - Needs Documentation
**Description**: A very high-tier medication that may be denied
- **Patient**: 33-year-old female, medium risk score (0.4)
- **Service**: Secukinumab (Tier 5, requires PA, high cost, step therapy)
- **Expected Result**: **DENIED** - "Service not medically necessary under plan criteria"
- **Appeal Prediction**: **MEDIUM RISK** - Appeal recommended with 65% confidence

## Rule Engine Logic

### Approval Criteria
1. **Emergency Services**: Always approved regardless of other factors
2. **Low Risk + Low Tier + Low Cost**: Auto-approved
3. **High Cost + High Risk**: Requires documentation
4. **Step Therapy Required**: Needs documentation to prove step therapy completion
5. **High Tier (≥4)**: Requires additional review
6. **State-Specific Rules**: Florida requires PA documentation for certain services

### Denial Criteria
1. **High Cost + No PA + Low Risk + Very High Tier**: Likely denied
2. **Services not meeting medical necessity criteria**

### Appeal Prediction Logic
- **High Risk**: Confidence ≥ 80%
- **Medium Risk**: Confidence 60-79%
- **Low Risk**: Confidence < 60%
- **Factors**: Patient risk score, service cost, tier level, prior denials

## API Endpoints Tested

### Primary Endpoints
- `POST /requests` - Submit new service request
- `GET /api/doctor-requests` - Retrieve doctor's requests
- `POST /eligibility-check` - Re-evaluate request
- `POST /appeals-predict` - Get ML appeal prediction

### Real-time Events
- `new_request` - Emitted when new request is created
- `request_updated` - Emitted when request status changes

## Error Handling Scenarios

### Input Validation Errors
- Missing patient ID: Returns 400 error
- Invalid item ID: Returns 404 error
- Server errors: Returns 500 error

### Expected Error Responses
```json
{
  "error": "patient.patient_id is required"
}
```

## Testing Instructions

### 1. Basic Functionality Test
1. Use Scenario 001 (Approved Statin) to test basic approval flow
2. Verify the request is created with status "Approved"
3. Check real-time event emission

### 2. Documentation Required Test
1. Use Scenario 002 (GLP-1 with Step Therapy) to test documentation requirement
2. Verify status is "Needs Docs"
3. Test document upload functionality

### 3. Denial and Appeal Test
1. Use Scenario 007 (Denied Tirzepatide) to test denial flow
2. Verify ML model predicts appeal recommendation
3. Test appeal submission process

### 4. Emergency Service Test
1. Use Scenario 006 (Emergency Cardiac) to test emergency approval
2. Verify immediate approval regardless of other factors

### 5. State-Specific Rules Test
1. Use Scenario 011 (Florida PA requirement) to test state rules
2. Verify Florida-specific documentation requirement

## Performance Considerations

### Response Times
- Standard request processing: < 2 seconds
- ML appeal prediction: < 3 seconds
- Real-time event emission: < 500ms

### Load Testing
- Test with 100+ concurrent requests
- Verify database performance under load
- Monitor real-time event delivery

## Integration Points

### External Services
- UHC Rules Engine (local + optional LLM)
- ML Model for Appeal Prediction
- Real-time WebSocket communication

### Database Operations
- Patient data upsert
- Request creation
- Service status updates
- Appeal tracking

## Monitoring and Logging

### Key Metrics
- Request approval rates by service type
- Appeal prediction accuracy
- Response time percentiles
- Error rates by endpoint

### Log Events
- Request creation with approval status
- ML prediction results
- Admin status changes
- Appeal submissions

## Maintenance Notes

### Rule Updates
- Update `uhc_rules.json` for new coverage rules
- Test all scenarios after rule changes
- Verify ML model accuracy with new data

### Model Updates
- Retrain ML model with new appeal data
- Update feature engineering as needed
- Validate prediction accuracy

This comprehensive test suite ensures the Doctor Dashboard handles all major scenarios correctly and provides reliable service to healthcare providers.
