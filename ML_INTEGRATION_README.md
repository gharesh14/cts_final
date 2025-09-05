# ML Model Integration for Appeal Risk Assessment

## Overview
This document describes the integration of a trained Machine Learning model for automatic appeal risk assessment in the MedCare Pro insurance system.

## Features Implemented

### 1. ML Model Integration
- **Automatic Loading**: ML model (`appeal_risk_model.pkl1`) is loaded at application startup
- **Fallback Support**: If ML model fails to load, system falls back to heuristic-based prediction
- **Real-time Prediction**: ML predictions run automatically when requests are denied

### 2. Database Schema Updates
- Added `appeal_recommended` (boolean) field to `ServiceRequested` table
- Enhanced `appeal_confidence` and `appeal_risk` fields with ML predictions
- Automatic storage of ML results when predictions are made

### 3. New API Endpoints

#### `/appealed-requests` (GET)
- Returns all requests where `appeal_recommended=True`
- Used by admin dashboard for global appeal overview
- Includes patient info, service details, and ML confidence scores

#### `/appeals/<doctor_id>` (GET)
- Returns appealed requests for a specific doctor
- Used by doctor dashboard to show their patients' appealed requests
- Currently returns all appealed requests (doctor filtering to be implemented)

#### `/appeals-predict` (POST)
- Accepts `item_id` and returns ML prediction
- Can be called manually to re-run ML analysis
- Returns appeal recommendation, confidence score, and risk level

### 4. Frontend Updates

#### Doctor Dashboard
- **Appeals Tab**: Shows all denied requests flagged for appeal by ML model
- **ML Insights**: Displays confidence scores and risk levels
- **Action Buttons**: Submit appeal, view details, and ML prediction info
- **Real-time Updates**: Socket.IO integration for live appeal status updates

#### Admin Dashboard
- **Global Appeal View**: Shows all appealed requests across all doctors
- **ML Statistics**: Displays appeal counts by risk level (High/Medium/Low)
- **Re-run ML**: Admins can manually trigger ML predictions
- **Risk Assessment**: Visual indicators for appeal confidence and risk levels

### 5. ML Model Features

#### Input Features
- **Patient Data**: Age, gender, risk score
- **Service Data**: Tier, requires PA, high cost, step therapy, estimated cost
- **Request Data**: Prior denials, deductible, coinsurance

#### Output Predictions
- **Appeal Recommendation**: Boolean (True/False)
- **Confidence Score**: Float (0.0 - 1.0)
- **Risk Level**: Categorical (High/Medium/Low)

#### Fallback Heuristics
When ML model is unavailable, system uses:
- Patient risk score > 0.7: +0.3 points
- Patient age > 65: +0.2 points
- High-cost service: +0.2 points
- High tier (≥4): +0.2 points
- Requires PA: +0.1 points
- Prior denials > 0: +0.2 points
- Appeal threshold: ≥0.5 points

## Technical Implementation

### Backend (Flask)
```python
# ML Model Loading
try:
    with open('appeal_risk_model.pkl1', 'rb') as f:
        ml_model_data = pickle.load(f)
        appeal_model = ml_model_data.get('model')
        appeal_scaler = ml_model_data.get('scaler')
    ML_MODEL_AVAILABLE = True
except Exception as e:
    ML_MODEL_AVAILABLE = False
    # Fallback to heuristics
```

### Database Schema
```sql
ALTER TABLE service_requested 
ADD COLUMN appeal_recommended BOOLEAN DEFAULT FALSE;
```

### API Response Format
```json
{
  "appealed_requests": [
    {
      "item_id": "I1234567890",
      "patient_name": "John Smith",
      "service_name": "MRI Scan",
      "appeal_recommended": true,
      "appeal_confidence": 0.85,
      "appeal_risk": "High"
    }
  ]
}
```

## Error Handling

### ML Model Failures
- Graceful fallback to heuristic-based prediction
- Logging of ML errors for debugging
- User notification when ML predictions are unavailable

### Database Errors
- Transaction rollback on failures
- Detailed error logging
- User-friendly error messages

### API Failures
- HTTP status codes for different error types
- JSON error responses with descriptive messages
- Frontend error handling and user notifications

## Security Considerations

### Data Privacy
- Patient data is processed locally (no external ML API calls)
- Sensitive information is not logged or exposed
- Database access is restricted to authenticated users

### Input Validation
- All API inputs are validated and sanitized
- ML model inputs are type-checked and normalized
- SQL injection protection via SQLAlchemy ORM

## Performance Optimizations

### ML Model Caching
- Model loaded once at startup
- Feature preprocessing optimized for batch operations
- Prediction results cached in database

### Database Queries
- Efficient joins between related tables
- Indexed fields for fast lookups
- Pagination support for large datasets

### Real-time Updates
- Socket.IO for live dashboard updates
- Debounced API calls to prevent spam
- Efficient event emission for status changes

## Usage Examples

### 1. Automatic ML Prediction
When a request is denied, the system automatically:
1. Extracts features from patient, service, and request data
2. Runs ML model prediction
3. Stores results in database
4. Updates frontend dashboards in real-time

### 2. Manual ML Re-run
Admins can manually trigger ML predictions:
```javascript
// Frontend call
fetch('/appeals-predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ item_id: 'I1234567890' })
});
```

### 3. Appeal Submission
Doctors can submit appeals for ML-recommended cases:
```javascript
// Frontend call
fetch('/appeals', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        item_id: 'I1234567890',
        appeal_notes: 'Additional clinical justification...'
    })
});
```

## Future Enhancements

### 1. Doctor-specific Filtering
- Implement proper doctor_id filtering in appeals endpoints
- User authentication and role-based access control
- Doctor-specific appeal dashboards

### 2. Advanced ML Features
- Model retraining with new data
- Feature importance analysis
- A/B testing for different ML models

### 3. Analytics Dashboard
- Appeal success rate tracking
- ML model performance metrics
- Cost-benefit analysis of appeals

### 4. Integration Enhancements
- FHIR compliance for healthcare standards
- External appeal system integration
- Automated appeal submission workflows

## Troubleshooting

### Common Issues

#### ML Model Not Loading
- Check if `appeal_risk_model.pkl1` exists in project root
- Verify scikit-learn and numpy are installed
- Check console logs for detailed error messages

#### Database Schema Issues
- Run `db.create_all()` to create new columns
- Check database connection and permissions
- Verify table structure matches model definitions

#### Frontend Not Updating
- Check browser console for JavaScript errors
- Verify Socket.IO connection is established
- Check network tab for failed API calls

### Debug Mode
Enable debug logging by setting:
```python
app.debug = True
```

### Log Files
Check console output for:
- ML model loading status
- Prediction errors and fallbacks
- Database operation results
- API endpoint access logs

## Support

For technical support or questions about the ML integration:
1. Check console logs for error messages
2. Verify all dependencies are installed
3. Test ML model loading independently
4. Review database schema and connections

## Dependencies

### Required Python Packages
- Flask 2.3.3+
- scikit-learn 1.3.0+
- numpy 1.24.3+
- pandas 2.0.3+
- mysql-connector-python 8.1.0+

### ML Model Format
- Pickle file with 'model' and 'scaler' keys
- scikit-learn RandomForestClassifier compatible
- StandardScaler for feature normalization
- Feature list for input validation

