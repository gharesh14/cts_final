// Doctor Dashboard JavaScript
document.addEventListener('DOMContentLoaded', function () {
    console.log('Doctor Dashboard loaded successfully');

    // Initialize the dashboard
    initializeDashboard();

    // Set up event listeners
    setupEventListeners();

    // Load initial data
    loadInitialData();

    // 🔄 Auto-refresh requests every 15 seconds
    setInterval(fetchRequestsFromBackend, 15000);

    // Connect to SocketIO for real-time updates
    connectToSocket();
});

// Connect to SocketIO for real-time updates
function connectToSocket() {
    const socket = io();

    // Listen for request status updates
    socket.on('request_updated', function (data) {
        console.log('Request status updated:', data);

        // Update the specific request in the table
        updateRequestStatus(data.id, data.status);

        // Show notification with admin notes if available
        let notificationMessage = `Request ${data.id} status updated to: ${data.status}`;
        if (data.admin_notes) {
            notificationMessage += ` - Admin Notes: ${data.admin_notes}`;
        }
        if (data.rule_reason) {
            notificationMessage += ` - Rule Reason: ${data.rule_reason}`;
        }
        showNotification(notificationMessage, 'info');
    });

    // Listen for new requests created (to reflect initial Approved/Denied/Needs Docs quickly)
    socket.on('new_request', function (data) {
        console.log('New request event:', data);
        showNotification(`Request submitted: ${data.service} - Status: ${data.status}${data.ruleReason ? ' (' + data.ruleReason + ')' : ''}`, 'success');
        fetchRequestsFromBackend();
    });

    // FIXED: Listen for new appeals
    socket.on('new_appeal', function (data) {
        console.log('New appeal submitted:', data);
        showNotification(`Appeal submitted for request ${data.request_id}`, 'success');
        // Refresh appeals list if on appeals tab
        if (document.getElementById('appeals').classList.contains('active')) {
            loadAppealedRequests();
        }
    });

    // FIXED: Listen for ML prediction updates
    socket.on('ml_prediction_updated', function (data) {
        console.log('ML prediction updated:', data);
        showNotification(`ML prediction updated: ${data.message}`, 'info');
    });

    // FIXED: Listen for document uploads
    socket.on('document_uploaded', function (data) {
        console.log('Document uploaded:', data);
        showNotification(`Document uploaded: ${data.file_name}`, 'success');
    });

    // Listen for appeal decision updates
    socket.on('appeal_decision_updated', function (data) {
        console.log('Appeal decision updated:', data);
        showNotification(`Your appeal ${data.appeal_id} has been ${data.decision.toLowerCase()}`, 'info');
        // Refresh appeals table
        loadAppealedRequests();
    });
}

// Dashboard initialization
function initializeDashboard() {
    // Show default tab
    showTab('submit-request');

    // Initialize file upload functionality
    initializeFileUpload();

    // Set up drag and drop for modal
    setupModalDragAndDrop();
}

// Event listeners setup
function setupEventListeners() {
    // Tab navigation
    const navTabs = document.querySelectorAll('.nav-tab');
    navTabs.forEach(tab => {
        tab.addEventListener('click', function () {
            const tabName = this.getAttribute('data-tab');
            showTab(tabName);
        });
    });

    // Form submission
    const requestForm = document.getElementById('request-form');
    if (requestForm) {
        console.log('Adding submit event listener to request form');
        requestForm.addEventListener('submit', handleFormSubmission);
        
        // Also prevent any default form submission behavior
        requestForm.addEventListener('submit', function(e) {
            e.preventDefault();
            e.stopPropagation();
            console.log('Default form submission prevented');
        });
    }

    // Patient search
    const patientSearch = document.getElementById('patient-search');
    if (patientSearch) {
        patientSearch.addEventListener('input', handlePatientSearch);
    }

    // Service name change handler
    const serviceNameInput = document.getElementById('service-name');
    if (serviceNameInput) {
        serviceNameInput.addEventListener('input', handleServiceNameChange);
    }

    // Modal file input change
    const modalFileInput = document.getElementById('modal-file-input');
    if (modalFileInput) {
        modalFileInput.addEventListener('change', handleModalFileSelection);
    }

    // Modal close on outside click
    window.addEventListener('click', function (event) {
        if (event.target.classList.contains('modal')) {
            event.target.style.display = 'none';
        }
    });
}

// Tab navigation
function showTab(tabName) {
    // Hide all tab contents
    const tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(content => {
        content.classList.remove('active');
    });

    // Remove active class from all tabs
    const navTabs = document.querySelectorAll('.nav-tab');
    navTabs.forEach(tab => {
        tab.classList.remove('active');
    });

    // Show selected tab content
    const selectedTab = document.getElementById(tabName);
    if (selectedTab) {
        selectedTab.classList.add('active');
    }

    // Add active class to selected tab
    const selectedNavTab = document.querySelector(`[data-tab="${tabName}"]`);
    if (selectedNavTab) {
        selectedNavTab.classList.add('active');
    }

    // Load tab-specific data
    loadTabData(tabName);
}

// Load tab-specific data
function loadTabData(tabName) {
    switch (tabName) {
        case 'my-requests':
            fetchRequestsFromBackend();
            break;
        case 'appeals':
            loadAppealsList();
            break;
    }
}

// Load initial data
function loadInitialData() {
    fetchRequestsFromBackend();
    loadAppealedRequests();
    // Mark initial load as complete
    window.initialLoadComplete = true;
}

// Fetch requests from backend and populate the table
async function fetchRequestsFromBackend() {
    try {
        const response = await fetch('/api/doctor-requests');
        if (!response.ok) {
            throw new Error('Failed to fetch doctor requests');
        }
        const data = await response.json();
        // Normalize minimal fields required by simplified table and actions
        window.allRequests = (data.requests || []).map(r => ({
            id: r.itemId || r.id || r.requestId,
            itemId: r.itemId || r.id || r.requestId,
            requestId: r.requestId,
            patientId: r.patientId,
            patientName: r.patientName,
            diagnosis: r.diagnosis || r.diagnosisCode,
            status: r.overallStatus || r.status,
            ruleReason: r.ruleReason,
            // carry through any appeal fields if backend provides them
            appealLevelPercentage: r.appealLevelPercentage || r.appeal_level_percentage,
            appeal_level: r.appeal_level,
            appealConfidence: r.appealConfidence,
            appealRecommended: r.appealRecommended,
            appeal_allowed: r.appeal_allowed,
            appealRisk: r.appealRisk
        }));
        loadRequestsTable(window.allRequests);
    } catch (error) {
        console.error('Error fetching doctor requests:', error);
        showNotification('Failed to load your requests. Please try again later.', 'error');
    }
}

// Load requests table
function loadRequestsTable(requests) {
    const tbody = document.getElementById('requests-tbody');
    if (!tbody || !requests) return;

    tbody.innerHTML = '';

    requests.forEach(request => {
        const row = createRequestRow(request);
        tbody.appendChild(row);
    });
}

// Load appealed requests for the appeals tab
async function loadAppealedRequests() {
    try {
        const response = await fetch('/api/doctor-appeals'); // Updated endpoint
        if (!response.ok) {
            throw new Error('Failed to fetch appealed requests');
        }
        const data = await response.json();
        // Backend returns { appeals: [...] }
        loadAppealsTable(data.appeals || []);
    } catch (error) {
        console.error('Error fetching appealed requests:', error);
        // Don't show error notification on initial load, just log it
        if (window.initialLoadComplete) {
            showNotification('Failed to load appealed requests. Please try again later.', 'error');
        }
        // Load empty table on error
        loadAppealsTable([]);
    }
}

// Load appeals table
function loadAppealsTable(appeals) {
    const tbody = document.getElementById('appeals-tbody');
    if (!tbody || !appeals) return;

    // Store appeals globally for access by other functions
    window.allAppeals = appeals;

    tbody.innerHTML = '';

    if (appeals.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="8" class="no-data">
                    <div class="no-data-content">
                        <i class="fas fa-check-circle"></i>
                        <p>No appealed requests found</p>
                        <small>All your requests are either approved or don't require appeals</small>
                    </div>
                </td>
            </tr>
        `;
        return;
    }

    appeals.forEach(appeal => {
        const row = createAppealRow(appeal);
        tbody.appendChild(row);
    });
}

// Create appeal table row
function createAppealRow(appeal) {
    const row = document.createElement('tr');

    const appealLevel = appeal.appealLevelPercentage || '0.0%';
    const originalStatus = appeal.originalStatus || 'Denied';
    const appealStatus = appeal.appeal_status || 'Submitted';
    const appealOutcome = appeal.appeal_outcome || 'Pending';

    // Determine status styling
    let statusClass = 'status-pending';
    let statusText = appealStatus;
    
    if (appealOutcome === 'Approved') {
        statusClass = 'status-approved';
        statusText = 'Approved';
    } else if (appealOutcome === 'Denied') {
        statusClass = 'status-denied';
        statusText = 'Denied';
    } else if (appealStatus === 'Under Review') {
        statusClass = 'status-reviewing';
        statusText = 'Under Review';
    }

    row.innerHTML = `
        <td><strong>${appeal.appeal_id || appeal.id || appeal.requestId}</strong></td>
        <td>
            <div class="patient-info">
                <strong>${appeal.patient_name || appeal.patientId || 'N/A'}</strong><br>
                <small>Request: ${appeal.request_id || appeal.requestId}</small>
            </div>
        </td>
        <td>
            <div class="service-info">
                <strong>${appeal.service_name || appeal.service || 'N/A'}</strong>
            </div>
        </td>
        <td>${originalStatus}</td>
        <td>
            <span class="confidence-score">${appealLevel}</span>
        </td>
        <td>
            <span class="status-chip ${statusClass}">${statusText}</span>
        </td>
        <td>${appeal.created_at || appeal.createdAt || ''}</td>
        <td>
            <div class="action-buttons">
                <button class="btn btn-secondary btn-sm" onclick="viewAppealDetails('${appeal.appeal_id || appeal.id}')">
                    <i class="fas fa-eye"></i> View
                </button>
            </div>
        </td>
    `;

    return row;
}

// Submit appeal for a denied request
async function submitAppeal(itemId) {
    try {
        const appealNotes = prompt('Please provide appeal notes:');
        if (!appealNotes) return;

        const response = await fetch('/appeals', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                item_id: itemId,
                appeal_notes: appealNotes
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to submit appeal');
        }

        const data = await response.json();
        showNotification(`Appeal submitted successfully! Appeal ID: ${data.appeal_id}`, 'success');

        // Refresh the appeals table and requests table
        loadAppealedRequests();
        await fetchRequestsFromBackend();

    } catch (error) {
        console.error('Error submitting appeal:', error);
        showNotification(`Failed to submit appeal: ${error.message}`, 'error');
    }
}

// View details of a request
function viewDetails(itemId) {
    // Find the request in the allRequests array
    const request = window.allRequests?.find(r => r.requestId === itemId);
    if (request) {
        showRequestDetailsModal(request);
    } else {
        showNotification('Request details not found', 'error');
    }
}

// View details of an appeal
function viewAppealDetails(appealId) {
    // Find the appeal in the appeals array
    const appeal = window.allAppeals?.find(a => a.appeal_id === appealId || a.id === appealId);
    if (appeal) {
        showAppealDetailsModal(appeal);
    } else {
        showNotification('Appeal details not found', 'error');
    }
}

// Show appeal details modal
function showAppealDetailsModal(appeal) {
    // Create modal if it doesn't exist
    let modal = document.getElementById('appeal-details-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'appeal-details-modal';
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>Appeal Details</h3>
                    <span class="close" onclick="closeAppealDetailsModal()">&times;</span>
                </div>
                <div class="modal-body" id="appeal-details-body">
                    <!-- Content will be populated here -->
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }

    // Populate modal content
    const modalBody = document.getElementById('appeal-details-body');
    modalBody.innerHTML = `
        <div class="appeal-details">
            <div class="detail-row">
                <label>Appeal ID:</label>
                <span>${appeal.appeal_id || appeal.id}</span>
            </div>
            <div class="detail-row">
                <label>Request ID:</label>
                <span>${appeal.request_id || appeal.requestId}</span>
            </div>
            <div class="detail-row">
                <label>Patient:</label>
                <span>${appeal.patient_name || appeal.patientId}</span>
            </div>
            <div class="detail-row">
                <label>Service:</label>
                <span>${appeal.service_name || appeal.service}</span>
            </div>
            <div class="detail-row">
                <label>Original Status:</label>
                <span>${appeal.originalStatus || 'Denied'}</span>
            </div>
            <div class="detail-row">
                <label>Appeal Status:</label>
                <span class="status-chip ${getAppealStatusClass(appeal)}">${appeal.appeal_status || 'Submitted'}</span>
            </div>
            <div class="detail-row">
                <label>Appeal Outcome:</label>
                <span class="status-chip ${getAppealOutcomeClass(appeal)}">${appeal.appeal_outcome || 'Pending'}</span>
            </div>
            <div class="detail-row">
                <label>Appeal Reason:</label>
                <div class="reason-text">${appeal.appeal_reason || 'No reason provided'}</div>
            </div>
            <div class="detail-row">
                <label>Submitted:</label>
                <span>${appeal.created_at || appeal.createdAt || 'Unknown'}</span>
            </div>
            ${appeal.admin_notes ? `
                <div class="detail-row">
                    <label>Admin Notes:</label>
                    <div class="admin-notes">${appeal.admin_notes}</div>
                </div>
            ` : ''}
        </div>
    `;

    modal.style.display = 'block';
}

// Get appeal status CSS class
function getAppealStatusClass(appeal) {
    const status = appeal.appeal_status || 'Submitted';
    switch (status) {
        case 'Submitted': return 'status-pending';
        case 'Under Review': return 'status-reviewing';
        case 'Completed': return 'status-approved';
        default: return 'status-pending';
    }
}

// Get appeal outcome CSS class
function getAppealOutcomeClass(appeal) {
    const outcome = appeal.appeal_outcome || 'Pending';
    switch (outcome) {
        case 'Approved': return 'status-approved';
        case 'Denied': return 'status-denied';
        case 'Pending': return 'status-pending';
        default: return 'status-pending';
    }
}

// Close appeal details modal
function closeAppealDetailsModal() {
    const modal = document.getElementById('appeal-details-modal');
    if (modal) {
        modal.style.display = 'none';
    }
}

// Update a specific request status in the table
function updateRequestStatus(itemId, newStatus) {
    const tbody = document.getElementById('requests-tbody');
    if (!tbody) return;

    const rows = tbody.getElementsByTagName('tr');
    for (let row of rows) {
        const idCell = row.querySelector('td:first-child strong');
        if (idCell && idCell.textContent === itemId) {
            const statusCell = row.querySelector('.status-chip');
            if (statusCell) {
                // Remove existing status classes
                statusCell.classList.remove('status-approved', 'status-denied', 'status-needs-docs');

                // Add new status class
                const statusClass = getStatusClass(newStatus);
                statusCell.classList.add(statusClass);

                // Update status text
                statusCell.textContent = newStatus;
            }
            break;
        }
    }
}

// Create request table row
function createRequestRow(request) {
    const row = document.createElement('tr');

    const itemId = request.itemId || request.id || request.requestId;
    const diagnosisText = request.diagnosis || request.diagnosisCode || '';
    const statusText = request.status || request.overallStatus || '';
    const statusClass = getStatusClass(statusText);

    row.setAttribute('data-item-id', itemId);
    row.innerHTML = `
        <td><strong>${itemId}</strong></td>
        <td>
            <div class="patient-info">
                <strong>${request.patientName || 'Unknown'}</strong><br>
                <small>ID: ${request.patientId || ''}</small>
            </div>
        </td>
        <td>
            <div class="diagnosis-info">
                <strong>${diagnosisText}</strong>
            </div>
        </td>
        <td>
            <div>
                <span class="status-chip ${statusClass}">${statusText}</span>
            </div>
        </td>
        <td>
            ${request.appealRisk ? `<span class="risk-chip">${request.appealRisk}</span>` : '<span class="text-muted">N/A</span>'}
        </td>
        <td>${request.ruleReason ? `${request.ruleReason}` : '<span class="text-muted">N/A</span>'}</td>
        <td>
            <div class="action-buttons">
                ${statusText === 'Needs Docs' ?
            `<button class="action-btn upload-docs-btn" onclick="openUploadModal('${itemId}')">
                        <i class="fas fa-upload"></i> Upload Docs
                    </button>` : ''}
                ${statusText === 'Denied' ? getAppealActionHtml({ ...request, itemId }) : ''}
            </div>
        </td>
    `;

    return row;
}

// Decide appeal action rendering based on backend prediction
function getAppealActionHtml(request) {
    // Show button based on 80% appeal threshold
    let score;
    if (typeof request.appeal_level === 'number') {
        score = request.appeal_level;
    } else if (typeof request.appealConfidence === 'number') {
        score = Math.round(request.appealConfidence * 100);
    } else if (typeof request.appealLevelPercentage === 'string') {
        const m = request.appealLevelPercentage.match(/([0-9]+\.?[0-9]*)%/);
        if (m) score = Math.round(parseFloat(m[1]));
    }
    
    if (typeof score === 'number' && score >= 80) {
        return `
            <button class="action-btn appeal-btn" onclick="openAppealModal('${request.itemId}')" title="Appeal this request (Score: ${score}%)">
                <i class="fas fa-gavel"></i> Appeal
            </button>
        `;
    }
    
    const msg = 'You cannot appeal for the request because probability of changing the result is low';
    return `<span class="text-muted" title="${msg}">${msg}${typeof score === 'number' ? ` (Score: ${score}%)` : ''}</span>`;
}

// Get status CSS class
function getStatusClass(status) {
    switch (status) {
        case 'Approved': return 'status-approved';
        case 'Denied': return 'status-denied';
        case 'Pending': return 'status-needs-docs';
        case 'Appealed': return 'status-denied'; // Appeals are styled like denied
        default: return '';
    }
}

// Load appeals list (currently stub)
async function loadAppealsList() {
    const appealsList = document.getElementById('appeals-list');
    if (!appealsList) return;

    try {
        const response = await fetch('/api/doctor-appeals');
        if (!response.ok) throw new Error('Failed to fetch appeals');
        const data = await response.json();

        if ((data.appeals || []).length === 0) {
            appealsList.innerHTML = '<p class="text-center text-muted">No appeals found.</p>';
            return;
        }

        appealsList.innerHTML = '';
        data.appeals.forEach(appeal => {
            const appealItem = document.createElement('div');
            appealItem.className = 'appeal-item';
            appealItem.innerHTML = `
                <div class="appeal-header">
                    <div class="appeal-info">
                        <h4>Appeal for Request ${appeal.request_id}</h4>
                        <p>Service: ${appeal.service_name}</p>
                    </div>
                    <span class="appeal-status">${appeal.appealLevelPercentage}</span>
                </div>
                <div class="appeal-details">
                    <p><strong>Original Status:</strong> ${appeal.originalStatus}</p>
                    <p><strong>Appeal Status:</strong> ${appeal.appeal_status || 'Submitted'}</p>
                    <p><strong>Created:</strong> ${appeal.created_at}</p>
                </div>
            `;
            appealsList.appendChild(appealItem);
        });

    } catch (error) {
        console.error('Error fetching appeals:', error);
        appealsList.innerHTML = '<p class="text-center text-danger">Failed to load appeals.</p>';
        showNotification('Failed to load appeals.', 'error');
    }
}

// Form handling
function handleFormSubmission(event) {
    event.preventDefault();
    event.stopPropagation();
    
    console.log('Form submission handled by handleFormSubmission');

    const requestData = {
        patientId: document.getElementById('patient-id').value,
        patientName: document.getElementById('patient-name').value,
        patientAge: document.getElementById('patient-age').value,
        patientGender: document.getElementById('patient-gender').value,
        patientState: document.getElementById('patient-state').value,
        icdCode: document.getElementById('diagnosis').value,
        serviceType: document.getElementById('service-type').value,
        serviceName: document.getElementById('service-name').value,
        planType: document.getElementById('plan-type').value,
        // Optional clinical fields only if present in DOM
        priorTherapies: document.getElementById('prior-therapies') ? document.getElementById('prior-therapies').value : '',
        hba1c: document.getElementById('hba1c') ? document.getElementById('hba1c').value : '',
        bmi: document.getElementById('bmi') ? document.getElementById('bmi').value : '',
        ldl: document.getElementById('ldl') ? document.getElementById('ldl').value : ''
    };

    if (!validateFormData(requestData)) {
        showNotification('Please fill in all required fields', 'error');
        return;
    }

    submitRequestToBackend(requestData);
}

// Validate form data
function validateFormData(data) {
    const requiredFields = [
        'patientId', 'patientName', 'patientAge', 'patientGender', 'patientState',
        'icdCode', 'serviceType', 'serviceName', 'planType'
    ];


    return requiredFields.every(field => {
        const value = data[field];
        return value !== null && value !== undefined && value !== '' && value.toString().trim() !== '';
    });
}

// Submit request
async function submitRequestToBackend(requestData) {
    console.log('submitRequestToBackend called with data:', requestData);
    
    const submitBtn = document.querySelector('.submit-btn');
    const originalText = submitBtn.innerHTML;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Submitting...';
    submitBtn.disabled = true;

    try {
        // Build minimal payload; backend rule engine handles risk, tier, cost, PA logic
        const requestPayload = {
            patient: {
                patient_id: requestData.patientId || `P${Date.now()}`,
                patient_name: requestData.patientName,
                age: parseInt(requestData.patientAge),
                gender: requestData.patientGender,
                state: requestData.patientState
            },
            request: {
                diagnosis: requestData.icdCode,
                plan_type: requestData.planType,
                prior_therapies: (requestData.priorTherapies || '')
                    .split(',')
                    .map(x => x.trim())
                    .filter(x => x.length > 0),
                hba1c: requestData.hba1c ? parseFloat(requestData.hba1c) : null,
                bmi: requestData.bmi ? parseFloat(requestData.bmi) : null,
                ldl: requestData.ldl ? parseFloat(requestData.ldl) : null
            },
            service: {
                service_type: requestData.serviceType,
                service_name: requestData.serviceName
            }

        };

        console.log('Making fetch request to /requests with payload:', requestPayload);
        
        const response = await fetch('/requests', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestPayload)
        });
        
        console.log('Fetch response received:', response);

        const result = await response.json();

        if (response.ok) {
            let notify = `Request submitted! Status: ${result.approval_status}`;
            // If denied and backend returned appeal decision, reflect in UI
            if (result.approval_status === 'Denied' && result.appeal_decision) {
                const ad = result.appeal_decision;
                if (ad.appeal_allowed) {
                    notify += ` | Appeal allowed (Score: ${ad.score}%)`;
                } else {
                    notify += ` | ${ad.message} (Score: ${ad.score}%)`;
                }
            }
            showNotification(notify, 'success');
            resetForm();
            await fetchRequestsFromBackend();
            // Automatically switch to My Requests tab after submission
            showTab('my-requests');
            // Refresh the requests table to show the new request
            await fetchRequestsFromBackend();
        } else {
            throw new Error(result.error || 'Failed to submit request');
        }
    } catch (error) {
        console.error('Error submitting request:', error);
        showNotification('Failed to submit request. Please try again.', 'error');
    } finally {
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
    }
}

// Reset form
function resetForm() {
    document.getElementById('request-form').reset();
    // Clear any additional form state if needed
}

// Patient search
function handlePatientSearch(event) {
    const query = event.target.value.toLowerCase();
    const suggestions = document.getElementById('patient-suggestions');

    if (query.length < 2) {
        suggestions.style.display = 'none';
        return;
    }

    // Mock patient data - in a real app, this would come from the backend
    const mockPatients = [
        { id: 'P001', name: 'John Doe', age: 45, gender: 'Male', state: 'NY' },
        { id: 'P002', name: 'Jane Smith', age: 32, gender: 'Female', state: 'CA' },
        { id: 'P003', name: 'Robert Johnson', age: 67, gender: 'Male', state: 'TX' }
    ];

    const filteredPatients = mockPatients.filter(patient =>
        patient.name.toLowerCase().includes(query) ||
        patient.id.toLowerCase().includes(query)
    );

    displayPatientSuggestions(filteredPatients);
}

// Display patient suggestions
function displayPatientSuggestions(patients) {
    const suggestions = document.getElementById('patient-suggestions');
    suggestions.innerHTML = '';

    if (patients.length === 0) {
        suggestions.style.display = 'none';
        return;
    }

    patients.forEach(patient => {
        const li = document.createElement('li');
        li.textContent = `${patient.name} (ID: ${patient.id}, Age: ${patient.age})`;
        li.onclick = () => selectPatient(patient);
        suggestions.appendChild(li);
    });

    suggestions.style.display = 'block';
}

// Select patient
function selectPatient(patient) {
    document.getElementById('patient-search').value = patient.id;
    document.getElementById('patient-name').value = patient.name;
    document.getElementById('patient-age').value = patient.age;
    document.getElementById('patient-gender').value = patient.gender;
    document.getElementById('patient-state').value = patient.state;

    document.getElementById('patient-suggestions').style.display = 'none';
}

// Handle service name change to auto-populate service code
function handleServiceNameChange(event) {
    const serviceName = event.target.value.toLowerCase();
    const serviceCodeInput = document.getElementById('service-code');

    // Map service names to common codes
    const serviceCodeMap = {
        'atorvastatin': 'NDC-12345-678-90',
        'rosuvastatin': 'NDC-12345-679-90',
        'simvastatin': 'NDC-12345-680-90',
        'pravastatin': 'NDC-12345-681-90',
        'lovastatin': 'NDC-12345-682-90',
        'semaglutide': 'NDC-12345-683-90',
        'dulaglutide': 'NDC-12345-684-90',
        'tirzepatide': 'NDC-12345-685-90',
        'evolocumab': 'NDC-12345-686-90',
        'alirocumab': 'NDC-12345-687-90',
        'adalimumab': 'NDC-12345-688-90',
        'etanercept': 'NDC-12345-689-90',
        'infliximab': 'NDC-12345-690-90',
        'ustekinumab': 'NDC-12345-691-90',
        'secukinumab': 'NDC-12345-692-90',
        'imatinib': 'NDC-12345-693-90',
        'rituximab': 'NDC-12345-694-90',
        'pembrolizumab': 'NDC-12345-695-90',
        'nivolumab': 'NDC-12345-696-90',
        'lenalidomide': 'NDC-12345-697-90'
    };

    // With simplified form, we no longer show service code; no-op
}

// Initialize file upload
function initializeFileUpload() {
    const fileInput = document.getElementById('file-input');
    const dropZone = document.getElementById('drop-zone');

    if (fileInput && dropZone) {
        fileInput.addEventListener('change', handleFileSelection);

        dropZone.addEventListener('dragover', function (e) {
            e.preventDefault();
            this.classList.add('drag-over');
        });

        dropZone.addEventListener('dragleave', function () {
            this.classList.remove('drag-over');
        });

        dropZone.addEventListener('drop', function (e) {
            e.preventDefault();
            this.classList.remove('drag-over');

            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                handleFileSelection({ target: fileInput });
            }
        });

        dropZone.addEventListener('click', function () {
            fileInput.click();
        });
    }
}

// Handle file selection
function handleFileSelection(event) {
    const files = event.target.files;
    const fileList = document.getElementById('file-list');

    if (!files || files.length === 0) return;

    fileList.innerHTML = '';

    Array.from(files).forEach(file => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <span class="file-name">${file.name}</span>
            <span class="file-size">(${formatFileSize(file.size)})</span>
            <button class="remove-file" onclick="removeFile(this)">×</button>
        `;
        fileList.appendChild(fileItem);
    });
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Remove file
function removeFile(button) {
    const fileItem = button.parentElement;
    fileItem.remove();

    // Clear the file input if no files are left
    const fileList = document.getElementById('file-list');
    if (fileList.children.length === 0) {
        document.getElementById('file-input').value = '';
    }
}

// Setup modal drag and drop
function setupModalDragAndDrop() {
    const modalDropZone = document.getElementById('modal-file-drop-zone');
    const modalFileInput = document.getElementById('modal-file-input');

    if (modalDropZone && modalFileInput) {
        modalDropZone.addEventListener('dragover', function (e) {
            e.preventDefault();
            this.classList.add('drag-over');
        });

        modalDropZone.addEventListener('dragleave', function () {
            this.classList.remove('drag-over');
        });

        modalDropZone.addEventListener('drop', function (e) {
            e.preventDefault();
            this.classList.remove('drag-over');

            if (e.dataTransfer.files.length > 0) {
                modalFileInput.files = e.dataTransfer.files;
                handleModalFileSelection({ target: modalFileInput });
            }
        });

        modalDropZone.addEventListener('click', function () {
            modalFileInput.click();
        });
    }
}

// Handle modal file selection
function handleModalFileSelection(event) {
    const files = event.target.files;
    const modalFileList = document.getElementById('modal-file-list');

    if (!files || files.length === 0) return;

    modalFileList.innerHTML = '';

    Array.from(files).forEach(file => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <span class="file-name">${file.name}</span>
            <span class="file-size">(${formatFileSize(file.size)})</span>
            <button class="remove-file" onclick="removeModalFile(this)">×</button>
        `;
        modalFileList.appendChild(fileItem);
    });
}

// Remove modal file
function removeModalFile(button) {
    const fileItem = button.parentElement;
    fileItem.remove();

    // Clear the modal file input if no files are left
    const modalFileList = document.getElementById('modal-file-list');
    if (modalFileList.children.length === 0) {
        document.getElementById('modal-file-input').value = '';
    }
}

// Open upload modal
function openUploadModal(itemId) {
    const modal = document.getElementById('upload-docs-modal');
    if (modal) {
        // Find the request details
        const request = window.allRequests?.find(r => r.itemId === itemId || r.id === itemId);
        if (request) {
            document.getElementById('upload-request-id').textContent = `Request: ${request.requestId}`;
            document.getElementById('upload-request-details').textContent =
                `Patient: ${request.patientName} | Service: ${request.diagnosis}`;
        }

        modal.style.display = 'block';
        modal.setAttribute('data-request-id', request?.itemId || itemId);
    }
}

// Close upload modal
function closeUploadModal() {
    const modal = document.getElementById('upload-docs-modal');
    if (modal) {
        modal.style.display = 'none';
        document.getElementById('modal-file-input').value = '';
        document.getElementById('modal-file-list').innerHTML = '';
    }
}

// Submit uploaded files
async function submitUploadedFiles() {
    const modal = document.getElementById('upload-docs-modal');
    const requestId = modal.getAttribute('data-request-id');
    const fileInput = document.getElementById('modal-file-input');

    if (!fileInput.files || fileInput.files.length === 0) {
        showNotification('Please select at least one file to upload', 'error');
        return;
    }

    const submitBtn = document.querySelector('.modal-submit-btn');
    const originalText = submitBtn.innerHTML;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Uploading...';
    submitBtn.disabled = true;

    try {
        // Upload files one by one
        const uploadPromises = Array.from(fileInput.files).map(async (file) => {
            const formData = new FormData();
            formData.append('item_id', requestId);
            formData.append('file', file);

            const response = await fetch('/eligibility-docs', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to upload file');
            }

            return await response.json();
        });

        const results = await Promise.all(uploadPromises);
        showNotification(`Successfully uploaded ${results.length} file(s)!`, 'success');
        closeUploadModal();

        // Refresh requests table to show updated status
        await fetchRequestsFromBackend();

    } catch (error) {
        console.error('Error uploading files:', error);
        showNotification(`Failed to upload documentation: ${error.message}`, 'error');
    } finally {
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
    }
}

// Open appeal modal
function openAppealModal(itemId) {
    const modal = document.getElementById('appeal-modal');
    if (modal) {
        // Find the request details
        const request = window.allRequests?.find(r => r.itemId === itemId || r.id === itemId);
        if (request) {
            document.getElementById('appeal-request-id').textContent = `Request: ${request.requestId}`;
            document.getElementById('appeal-request-details').textContent =
                `Patient: ${request.patientName} | Service: ${request.diagnosis}`;
        }

        modal.style.display = 'block';
        modal.setAttribute('data-request-id', request?.itemId || itemId);
    }
}

// Close appeal modal
function closeAppealModal() {
    const modal = document.getElementById('appeal-modal');
    if (modal) {
        modal.style.display = 'none';
        document.getElementById('appeal-notes').value = '';
        document.getElementById('appeal-evidence').value = '';
    }
}

// Submit appeal from modal
async function submitAppealFromModal() {
    const modal = document.getElementById('appeal-modal');
    const itemId = modal.getAttribute('data-request-id');
    const appealNotes = document.getElementById('appeal-notes').value;
    const appealEvidence = document.getElementById('appeal-evidence').value;

    if (!appealNotes.trim()) {
        showNotification('Please provide a reason for the appeal', 'error');
        return;
    }

    // Find the request to get the correct item_id
    const request = window.allRequests?.find(r => r.itemId === itemId || r.id === itemId);
    if (!request || !request.itemId) {
        showNotification('Request data not found. Please refresh and try again.', 'error');
        return;
    }

    const submitBtn = document.querySelector('.appeal-submit-btn');
    const originalText = submitBtn.innerHTML;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Submitting...';
    submitBtn.disabled = true;

    try {
        const response = await fetch('/appeals', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                item_id: request.itemId,
                appeal_notes: `${appealNotes}\n\nSupporting Evidence: ${appealEvidence || 'None provided'}`
            })
        });

        if (response.ok) {
            const data = await response.json();
            showNotification(`Appeal submitted successfully! Appeal ID: ${data.appeal_id}`, 'success');
            closeAppealModal();
            await fetchRequestsFromBackend();
        } else {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to submit appeal');
        }
    } catch (error) {
        console.error('Error submitting appeal:', error);
        showNotification(`Failed to submit appeal: ${error.message}`, 'error');
    } finally {
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
    }
}

// Note: showNotification function is now provided by notification.js

// Filter requests
function filterRequests() {
    const searchTerm = document.getElementById('request-search').value.toLowerCase();
    const statusFilter = document.getElementById('status-filter').value;

    let filteredRequests = window.allRequests || [];

    // Apply search filter
    if (searchTerm) {
        filteredRequests = filteredRequests.filter(request =>
            request.id.toLowerCase().includes(searchTerm) ||
            request.patientId.toLowerCase().includes(searchTerm) ||
            request.service.toLowerCase().includes(searchTerm) ||
            request.serviceType.toLowerCase().includes(searchTerm)
        );
    }

    // Apply status filter
    if (statusFilter !== 'all') {
        filteredRequests = filteredRequests.filter(request =>
            request.status.toLowerCase() === statusFilter.toLowerCase()
        );
    }

    // Update the table
    loadRequestsTable(filteredRequests);
}

// Export requests
function exportRequests() {
    // This would export the requests to CSV or Excel in a real application
    showNotification('Export functionality would be implemented here', 'info');
}

// Helper function to determine tier based on service name and type
function determineTier(serviceName, serviceType) {
    const name = serviceName.toLowerCase();
    const type = serviceType.toLowerCase();

    // High-tier medications (Tier 4-6)
    if (name.includes('evolocumab') || name.includes('alirocumab') || name.includes('pcsk9')) {
        return 4;
    }
    if (name.includes('adalimumab') || name.includes('etanercept') || name.includes('infliximab') ||
        name.includes('ustekinumab') || name.includes('secukinumab')) {
        return 5;
    }
    if (name.includes('imatinib') || name.includes('rituximab') || name.includes('pembrolizumab') ||
        name.includes('nivolumab') || name.includes('lenalidomide')) {
        return 6;
    }

    // GLP-1 medications (Tier 3)
    if (name.includes('semaglutide') || name.includes('dulaglutide') || name.includes('tirzepatide')) {
        return 3;
    }

    // Statins (Tier 1)
    if (name.includes('atorvastatin') || name.includes('rosuvastatin') || name.includes('simvastatin') ||
        name.includes('pravastatin') || name.includes('lovastatin')) {
        return 1;
    }

    // Default based on service type
    if (type === 'medication') return 2;
    if (type === 'imaging') return 3;
    if (type === 'procedure') return 4;
    if (type === 'dme') return 2;

    return 1; // Default tier
}

// Helper function to determine if step therapy is required
function determineStepTherapy(serviceName) {
    const name = serviceName.toLowerCase();

    // GLP-1 medications require step therapy
    if (name.includes('semaglutide') || name.includes('dulaglutide') || name.includes('tirzepatide')) {
        return true;
    }

    // Biologics require step therapy
    if (name.includes('adalimumab') || name.includes('etanercept') || name.includes('infliximab') ||
        name.includes('ustekinumab') || name.includes('secukinumab')) {
        return true;
    }

    return false;
}

// Infer an estimated cost range by service name/type to support rule engine fields
function inferEstimatedCost(serviceName, serviceType) {
    const name = (serviceName || '').toLowerCase();
    const type = (serviceType || '').toLowerCase();
    if (name.includes('evolocumab') || name.includes('alirocumab') || name.includes('pcsk9')) return 5500;
    if (name.includes('adalimumab') || name.includes('etanercept') || name.includes('infliximab') || name.includes('ustekinumab') || name.includes('secukinumab')) return 8000;
    if (name.includes('imatinib') || name.includes('rituximab') || name.includes('pembrolizumab') || name.includes('nivolumab') || name.includes('lenalidomide')) return 15000;
    if (name.includes('semaglutide') || name.includes('dulaglutide') || name.includes('tirzepatide')) return 1200;
    if (name.includes('atorvastatin') || name.includes('rosuvastatin') || name.includes('simvastatin') || name.includes('pravastatin') || name.includes('lovastatin')) return 150;
    if (type === 'imaging') return 2000;
    if (type === 'procedure') return 12000;
    if (type === 'dme') return 800;
    if (type === 'emergency') return 25000;
    return 1000;
}

// Update service name placeholder based on service type
function updateServiceNamePlaceholder() {
    const serviceType = document.getElementById('service-type').value;
    const serviceNameInput = document.getElementById('service-name');

    const placeholders = {
        'medication': 'e.g., Atorvastatin, Semaglutide, Evolocumab',
        'imaging': 'e.g., MRI Brain with Contrast, CT Scan',
        'procedure': 'e.g., Arthroscopic Knee Surgery, Cardiac Catheterization',
        'dme': 'e.g., Wheelchair - Standard, CPAP Machine',
        'emergency': 'e.g., Emergency Cardiac Catheterization'
    };

    if (serviceType && placeholders[serviceType]) {
        serviceNameInput.placeholder = placeholders[serviceType];
    } else {
        serviceNameInput.placeholder = 'Enter service name';
    }
}

// Global functions for HTML onclick handlers
window.openUploadModal = openUploadModal;
window.closeUploadModal = closeUploadModal;
window.submitUploadedFiles = submitUploadedFiles;
window.openAppealModal = openAppealModal;
window.closeAppealModal = closeAppealModal;
window.submitAppealFromModal = submitAppealFromModal;
window.removeFile = removeFile;
window.removeModalFile = removeModalFile;
window.filterRequests = filterRequests;
window.exportRequests = exportRequests;
window.updateServiceNamePlaceholder = updateServiceNamePlaceholder;
window.resetForm = resetForm;
window.viewAppealDetails = viewAppealDetails;
window.closeAppealDetailsModal = closeAppealDetailsModal;