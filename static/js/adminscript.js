// Admin Dashboard JavaScript
document.addEventListener('DOMContentLoaded', function() {
    console.log('Admin Dashboard loaded successfully');
    
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
    
    // Listen for new requests
    socket.on('new_request', function(data) {
        console.log('New request received:', data);
        
        // Add the new request to the table
        addNewRequestToTable(data);
        
        // Show notification
        showNotification(`New request received: ${data.service} for patient ${data.patientId}`, 'info');
    });
    
    // Listen for request status updates
    socket.on('request_updated', function(data) {
        console.log('Request status updated:', data);
        
        // Update the specific request in the table
        updateRequestStatus(data.id, data.status);
        
        // Show notification
        showNotification(`Request ${data.id} status updated to: ${data.status}`, 'info');
    });
    
    // FIXED: Listen for new appeals
    socket.on('new_appeal', function(data) {
        console.log('New appeal received:', data);
        showNotification(`New appeal submitted: ${data.service_name} for patient ${data.patient_name}`, 'warning');
        // Refresh appealed requests data
        loadAppealedRequests();
    });
    
    // FIXED: Listen for ML prediction updates
    socket.on('ml_prediction_updated', function(data) {
        console.log('ML prediction updated:', data);
        showNotification(`ML prediction updated: ${data.message}`, 'info');
        // Refresh appealed requests data
        loadAppealedRequests();
    });

    // Listen for appeal decision updates
    socket.on('appeal_decision_updated', function(data) {
        console.log('Appeal decision updated:', data);
        showNotification(`Appeal ${data.appeal_id} has been ${data.decision.toLowerCase()}`, 'info');
        // Refresh appeals table
        loadAppealedRequests();
    });
    
    // FIXED: Listen for document uploads
    socket.on('document_uploaded', function(data) {
        console.log('Document uploaded:', data);
        showNotification(`New document uploaded: ${data.file_name}`, 'info');
        // Refresh documentation list if on documentation tab
        if (document.getElementById('documentation').classList.contains('active')) {
            loadDocumentationList();
        }
    });
}

// Dashboard initialization
function initializeDashboard() {
    // Show default tab
    showTab('requests');
    
    // Initialize charts
    initializeCharts();
}

// Event listeners setup
function setupEventListeners() {
    // Tab navigation
    const navTabs = document.querySelectorAll('.nav-tab');
    navTabs.forEach(tab => {
        tab.addEventListener('click', function() {
            const tabName = this.getAttribute('data-tab');
            showTab(tabName);
        });
    });
    
    // Search and filter functionality
    const searchInput = document.getElementById('request-search');
    const statusFilter = document.getElementById('status-filter');
    
    if (searchInput) {
        searchInput.addEventListener('input', filterRequests);
    }
    
    if (statusFilter) {
        statusFilter.addEventListener('change', filterRequests);
    }
    
    // Modal close on outside click
    window.addEventListener('click', function(event) {
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
    switch(tabName) {
        case 'requests':
            fetchRequestsFromBackend();
            break;
        case 'appeals':
            loadAppealedRequests();
            break;
        case 'logs':
            loadAuditLogs();
            break;
    }
}

// Load initial data
function loadInitialData() {
    fetchRequestsFromBackend();
    loadAppealedRequests();
}

// Fetch requests from backend and populate the table
async function fetchRequestsFromBackend() {
    try {
        const response = await fetch('/api/request-data');
        if (!response.ok) {
            throw new Error('Failed to fetch request data');
        }
        const data = await response.json();
        window.allRequests = data.requests;
        loadRequestsTable(window.allRequests);
    } catch (error) {
        console.error('Error fetching request data:', error);
        showNotification('Failed to load request data. Please try again later.', 'error');
    }
}

// Add new request to table
function addNewRequestToTable(requestData) {
    const tbody = document.getElementById('requests-tbody');
    if (!tbody) return;
    
    // Check if request already exists in the table
    const existingRows = tbody.querySelectorAll('tr');
    for (let row of existingRows) {
        const idCell = row.querySelector('td:first-child strong');
        if (idCell && idCell.textContent === requestData.id) {
            return; // Request already exists
        }
    }
    
    // Add new request to the beginning of the table
    const row = createRequestRow(requestData);
    tbody.insertBefore(row, tbody.firstChild);
    
    // Update dashboard stats
    updateDashboardStats();
}

// Update a specific request status in the table
function updateRequestStatus(itemId, newStatus) {
    const tbody = document.getElementById('requests-tbody');
    if (!tbody) return;
    
    const rows = tbody.getElementsByTagName('tr');
    for (let row of rows) {
        const rowItemId = row.getAttribute('data-item-id');
        if (rowItemId === itemId) {
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
            
            // Update action buttons based on new status
            const actionCell = row.querySelector('.action-buttons');
            if (actionCell) {
                actionCell.innerHTML = getActionButtons(itemId, newStatus);
            }
            
            break;
        }
    }
    
    // Update dashboard stats
    updateDashboardStats();
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
    
    // Update dashboard stats
    updateDashboardStats();
}

// // Load appealed requests for the admin dashboard
// async function loadAppealedRequests() {
//     try {
//         const response = await fetch('/api/admin-appeals');
//         if (!response.ok) {
//             throw new Error('Failed to fetch appealed requests');
//         }
//         const data = await response.json();

//         // Check if the data is an object with an 'appeals' key, or if it's the array itself.
//         const appealsData = data.appeals || data;

//         updateAppealedRequestsStats(appealsData);
//         loadAppealsTable(appealsData);
//     } catch (error) {
//         console.error('Error fetching appealed requests:', error);
//         showNotification('Failed to load appealed requests data. Please try again later.', 'error');
//     }
// }

// // Load appeals table for admin dashboard
// function loadAppealsTable(appeals) {
//     const tbody = document.getElementById('appeals-tbody');
//     if (!tbody || !appeals) return;

//     // Store appeals globally for access by other functions
//     window.allAdminAppeals = appeals;

//     tbody.innerHTML = '';

//     if (appeals.length === 0) {
//         tbody.innerHTML = `
//             <tr>
//                 <td colspan="10" class="no-data">
//                     <div class="no-data-content">
//                         <i class="fas fa-check-circle"></i>
//                         <p>No appeals found</p>
//                         <small>All appeals have been processed or no new appeals submitted</small>
//                     </div>
//                 </td>
//             </tr>
//         `;
//         return;
//     }

//     appeals.forEach(appeal => {
//         const row = createAppealRow(appeal);
//         tbody.appendChild(row);
//     });
// }

// // Create appeal table row for admin dashboard
// function createAppealRow(appeal) {
//     const row = document.createElement('tr');

//     const appealStatus = appeal.appeal_status || 'Submitted';
//     const appealOutcome = appeal.appeal_outcome || 'Pending';

//     // Determine status styling
//     let statusClass = 'status-pending';
//     let statusText = appealStatus;

//     if (appealOutcome === 'Approved') {
//         statusClass = 'status-approved';
//         statusText = 'Approved';
//     } else if (appealOutcome === 'Denied') {
//         statusClass = 'status-denied';
//         statusText = 'Denied';
//     } else if (appealStatus === 'Under Review') {
//         statusClass = 'status-reviewing';
//         statusText = 'Under Review';
//     }

//     row.innerHTML = `
//         <td><strong>${appeal.appeal_id}</strong></td>
//         <td><strong>${appeal.request_id}</strong></td>
//         <td>
//             <div class="patient-info">
//                 <strong>${appeal.patient_name}</strong><br>
//                 <small>ID: ${appeal.patient_id}</small>
//             </div>
//         </td>
//         <td>
//             <div class="service-info">
//                 <strong>${appeal.service_name}</strong><br>
//                 <small>${appeal.service_type}</small>
//             </div>
//         </td>
//         <td><span class="status-chip status-denied">${appeal.originalStatus}</span></td>
//         <td>
//             <span class="confidence-score">${appeal.appealLevelPercentage}</span>
//         </td>
//         <td>
//             <div class="appeal-reason">${appeal.appeal_reason ? appeal.appeal_reason.substring(0, 50) + '...' : 'No reason provided'}</div>
//         </td>
//     `;

//     return row;
// }
// Approve an appeal
async function approveAppeal(appealId) {
    const adminNotes = prompt('Please provide admin notes for this approval (optional):');
    if (adminNotes === null) return; // User cancelled
    
    try {
        const response = await fetch(`/api/admin-appeals/${appealId}/decision`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                decision: 'Approved',
                admin_notes: adminNotes || ''
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to approve appeal');
        }

        const data = await response.json();
        showNotification(`Appeal ${appealId} has been approved!`, 'success');
        
        // Refresh the appeals table
        await loadAppealedRequests();
        
    } catch (error) {
        console.error('Error approving appeal:', error);
        showNotification(`Failed to approve appeal: ${error.message}`, 'error');
    }
}

// Deny an appeal
async function denyAppeal(appealId) {
    const adminNotes = prompt('Please provide reason for denial (required):');
    if (!adminNotes || adminNotes.trim() === '') {
        showNotification('Admin notes are required for denial', 'error');
        return;
    }
    
    try {
        const response = await fetch(`/api/admin-appeals/${appealId}/decision`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                decision: 'Denied',
                admin_notes: adminNotes
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to deny appeal');
        }

        const data = await response.json();
        showNotification(`Appeal ${appealId} has been denied.`, 'info');
        
        // Refresh the appeals table
        await loadAppealedRequests();
        
    } catch (error) {
        console.error('Error denying appeal:', error);
        showNotification(`Failed to deny appeal: ${error.message}`, 'error');
    }
}

// View appeal details
function viewAppealDetails(appealId) {
    const appeal = window.allAdminAppeals?.find(a => a.appeal_id === appealId);
    if (!appeal) {
        showNotification('Appeal details not found', 'error');
        return;
    }

    // Create modal if it doesn't exist
    let modal = document.getElementById('admin-appeal-details-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'admin-appeal-details-modal';
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>Appeal Details</h3>
                    <span class="close" onclick="closeAdminAppealDetailsModal()">&times;</span>
                </div>
                <div class="modal-body" id="admin-appeal-details-body">
                    <!-- Content will be populated here -->
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }

    // Populate modal content
    const modalBody = document.getElementById('admin-appeal-details-body');
    modalBody.innerHTML = `
        <div class="appeal-details">
            <div class="detail-row">
                <label>Appeal ID:</label>
                <span>${appeal.appeal_id}</span>
            </div>
            <div class="detail-row">
                <label>Request ID:</label>
                <span>${appeal.request_id}</span>
            </div>
            <div class="detail-row">
                <label>Patient:</label>
                <span>${appeal.patient_name} (${appeal.patient_id})</span>
            </div>
            <div class="detail-row">
                <label>Service:</label>
                <span>${appeal.service_name} (${appeal.service_type})</span>
            </div>
            <div class="detail-row">
                <label>Original Status:</label>
                <span>${appeal.originalStatus}</span>
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
                <label>Supporting Documents:</label>
                <div class="documentation-text">${appeal.appeal_documents || 'No documents provided'}</div>
            </div>
            <div class="detail-row">
                <label>Submitted:</label>
                <span>${appeal.created_at || 'Unknown'}</span>
            </div>
            ${appeal.reviewed_at ? `
                <div class="detail-row">
                    <label>Reviewed:</label>
                    <span>${appeal.reviewed_at}</span>
                </div>
            ` : ''}
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

// Close admin appeal details modal
function closeAdminAppealDetailsModal() {
    const modal = document.getElementById('admin-appeal-details-modal');
    if (modal) {
        modal.style.display = 'none';
    }
}

// Update appealed requests statistics in the dashboard
function updateAppealedRequestsStats(appealedRequests) {
    const totalAppeals = appealedRequests.length;
    const highRiskAppeals = appealedRequests.filter(req => req.appeal_risk === 'High').length;
    const mediumRiskAppeals = appealedRequests.filter(req => req.appeal_risk === 'Medium').length;
    const lowRiskAppeals = appealedRequests.filter(req => req.appeal_risk === 'Low').length;
    
    // Update appeals tab stats if available
    const appealsStatsContainer = document.querySelector('.appeals-stats');
    if (appealsStatsContainer) {
        appealsStatsContainer.innerHTML = `
            <div class="stat-card">
                <div class="stat-icon appeals">
                    <i class="fas fa-gavel"></i>
                </div>
                <div class="stat-content">
                    <h3>Total Appeals</h3>
                    <p class="stat-number">${totalAppeals}</p>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon high-risk">
                    <i class="fas fa-exclamation-triangle"></i>
                </div>
                <div class="stat-content">
                    <h3>High Risk</h3>
                    <p class="stat-number">${highRiskAppeals}</p>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon medium-risk">
                    <i class="fas fa-exclamation-circle"></i>
                </div>
                <div class="stat-content">
                    <h3>Medium Risk</h3>
                    <p class="stat-number">${mediumRiskAppeals}</p>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon low-risk">
                    <i class="fas fa-info-circle"></i>
                </div>
                <div class="stat-content">
                    <h3>Low Risk</h3>
                    <p class="stat-number">${lowRiskAppeals}</p>
                </div>
            </div>
        `;
    }
    
    // Update appeals table if on appeals tab
    const appealsTable = document.getElementById('appeals-tbody');
    if (appealsTable) {
        loadAppealsTable(appealedRequests);
    }
}

// Load appeals table with appealed requests
function loadAppealsTable(appeals) {
    const tbody = document.getElementById('appeals-tbody');
    if (!tbody || !appeals) return;
    
    tbody.innerHTML = '';
    
    if (appeals.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="8" class="no-data">
                    <div class="no-data-content">
                        <i class="fas fa-check-circle"></i>
                        <p>No appealed requests found</p>
                        <small>All requests are either approved or don't require appeals</small>
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

// Create appeal table row for admin dashboard
function createAppealRow(appeal) {
    const row = document.createElement('tr');
    
    const confidenceClass = appeal.appeal_confidence >= 0.8 ? 'high-confidence' : 
                           appeal.appeal_confidence >= 0.6 ? 'medium-confidence' : 'low-confidence';
    
    row.innerHTML = `
        <td><strong>${appeal.item_id}</strong></td>
        <td>
            <div class="patient-info">
                <strong>${appeal.patient_name}</strong><br>
                <small>ID: ${appeal.patient_id}</small><br>
                <small>${appeal.patient_age} yrs</small>
            </div>
        </td>
        <td>
            <div class="service-info">
                <strong>${appeal.service_name}</strong><br>
                <small>${appeal.service_type}</small>
            </div>
        </td>
        <td>${appeal.diagnosis}</td>
        <td><span class="status-chip status-denied">${appeal.approval_status}</span></td>
        <td>
            <span class="confidence-score ${confidenceClass}">
                ${(appeal.appeal_confidence * 100).toFixed(1)}%
            </span>
        </td>
        <td>
            <span class="risk-level ${appeal.appeal_risk.toLowerCase()}">
                ${appeal.appeal_risk}
            </span>
        </td>
    `;
    
    return row;
}

// Run ML prediction for a specific request
async function runMLPrediction(itemId) {
    try {
        const response = await fetch('/appeals-predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                item_id: itemId
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to run ML prediction');
        }
        
        const data = await response.json();
        showNotification(`ML Prediction: Appeal ${data.appeal_recommended ? 'Recommended' : 'Not Recommended'} (${data.confidence_percentage} confidence)`, 'success');
        
        // Refresh the appealed requests data
        loadAppealedRequests();
        
    } catch (error) {
        console.error('Error running ML prediction:', error);
        showNotification('Failed to run ML prediction. Please try again.', 'error');
    }
}

// View appeal details
function viewAppealDetails(itemId) {
    // Find the appeal in the appealed requests
    const appeal = window.appealedRequests?.find(a => a.item_id === itemId);
    if (appeal) {
        showAppealDetailsModal(appeal);
    } else {
        showNotification('Appeal details not found', 'error');
    }
}

// Create request table row
function createRequestRow(request) {
    const row = document.createElement('tr');
    
    const statusClass = getStatusClass(request.overallStatus);
    const riskClass = request.riskLevel;
    const itemId = request.itemId; // used for actions/API
    row.setAttribute('data-item-id', itemId || '');
    
    row.innerHTML = `
        <td><strong>${request.requestId}</strong></td>
        <td>
            <div class="patient-info">
                <strong>${request.patientName}</strong><br>
                <small>ID: ${request.patientId}</small><br>
                <small>${request.patientAge} yrs, ${request.patientGender}, ${request.patientState}</small>
            </div>
        </td>
        <td>
            <div class="diagnosis-info">
                <strong>${request.diagnosis}</strong><br>
                <small>Category: ${request.diagnosisCategory}</small>
            </div>
        </td>
        <td>
            <div class="plan-info">
                <strong>${request.planType}</strong><br>
                <small>Deductible: ${request.deductible}</small><br>
                <small>Coinsurance: ${request.coinsurance}</small><br>
                <small>OOP Max: ${request.outOfPocketMax}</small>
            </div>
        </td>
        <td>
            <div>
                <span class="status-chip ${statusClass}">${request.overallStatus}</span>
                ${request.ruleReason ? `<div class="text-muted" style="margin-top:4px;max-width:320px;white-space:normal;">Reason: ${request.ruleReason}</div>` : ''}
            </div>
        </td>
        <td>${request.totalServices}</td>
        
        <td>${request.timestamp}</td>
        <td>
            ${renderAppealGuidance(request)}
        </td>
    `;
    
    return row;
}

// Render appeal guidance for admin based on 80% threshold
function renderAppealGuidance(request) {
    // derive score from available fields
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
        return `<span>You can appeal this request again. (Score: ${score}%)</span>`;
    }
    return `<span class="text-muted">Doctor cannot appeal the request again (low probability)</span>`;
}

// Get action buttons based on request status
function getActionButtons(itemId, status) {
    let buttons = '';
    
    // Always show the rule engine button
    buttons += `
        <button class="action-btn rule-engine-btn" onclick="runRuleEngine('${itemId}')" title="Run Rule Engine">
            <i class="fas fa-robot"></i> AI Review
        </button>
    `;
    
    switch(status) {
        case 'Pending':
            buttons += `
                <button class="action-btn approve-btn" onclick="updateRequestStatusFromAdmin('${itemId}', 'Approved')">
                    <i class="fas fa-check"></i> Approve
                </button>
                <button class="action-btn deny-btn" onclick="updateRequestStatusFromAdmin('${itemId}', 'Denied')">
                    <i class="fas fa-times"></i> Deny
                </button>
                <button class="action-btn needs-docs-btn" onclick="updateRequestStatusFromAdmin('${itemId}', 'Needs Docs')">
                    <i class="fas fa-file-medical"></i> Needs Docs
                </button>
            `;
            break;
        case 'Needs Docs':
            buttons += `
                <button class="action-btn view-docs-btn" onclick="viewDocumentation('${itemId}')">
                    <i class="fas fa-eye"></i> View Docs
                </button>
            `;
            break;
        case 'Denied':
            buttons += `
                <button class="action-btn view-appeal-btn" onclick="viewAppeal('${itemId}')">
                    <i class="fas fa-gavel"></i> View Appeal
                </button>
            `;
            break;
        default:
            buttons += `<span class="text-muted">No actions</span>`;
    }
    
    return buttons;
}

// Get status CSS class
function getStatusClass(status) {
    switch(status) {
        case 'Approved': return 'status-approved';
        case 'Denied': return 'status-denied';
        case 'Needs Docs': return 'status-needs-docs';
        case 'Appealed': return 'status-denied'; // Appeals are styled like denied
        default: return '';
    }
}

// Update request status from admin
async function updateRequestStatusFromAdmin(itemId, newStatus) {
    try {
        const response = await fetch(`/api/requests/${itemId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status: newStatus })
        });
        
        if (response.ok) {
            showNotification(`Request ${itemId} status updated to ${newStatus}`, 'success');
            updateRequestStatus(itemId, newStatus);
        } else {
            throw new Error('Failed to update request status');
        }
    } catch (error) {
        console.error('Error updating request status:', error);
        showNotification('Failed to update request status. Please try again.', 'error');
    }
}

// Run rule engine for a specific request
async function runRuleEngine(itemId) {
    try {
        const response = await fetch(`/api/rule-engine/${itemId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        if (response.ok) {
            const result = await response.json();
            showNotification(`Rule engine result: ${result.new_status} - ${result.rule_reason}`, 'success');
            updateRequestStatus(itemId, result.new_status);
        } else {
            throw new Error('Failed to run rule engine');
        }
    } catch (error) {
        console.error('Error running rule engine:', error);
        showNotification('Failed to run rule engine. Please try again.', 'error');
    }
}

// Load appeals list
async function loadAppealsList() {
    const appealsTbody = document.getElementById('appeals-tbody');
    if (!appealsTbody) return;
    
    try {
        const response = await fetch('/api/appeals');
        if (!response.ok) throw new Error('Failed to fetch appeals');
        const data = await response.json();
        
        if ((data.appeals || []).length === 0) {
            appealsTbody.innerHTML = '<tr><td colspan="10" class="text-center text-muted">No appeals found.</td></tr>';
            return;
        }
        
        appealsTbody.innerHTML = '';
        data.appeals.forEach(appeal => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><strong>${appeal.id}</strong></td>
                <td>${appeal.requestId}</td>
                <td>${appeal.patientId}</td>
                <td>${appeal.service}</td>
                <td>${appeal.originalStatus}</td>
                <td>${appeal.appealLevelPercentage}</td>
                <td>${appeal.createdAt}</td>
                <td>
                    <div class="action-buttons">
                        <button class="action-btn view-appeal-btn" onclick="viewAppeal('${appeal.requestId}')">
                            <i class="fas fa-eye"></i> View
                        </button>
                    </div>
                </td>
            `;
            appealsTbody.appendChild(row);
        });
        
    } catch (error) {
        console.error('Error fetching appeals:', error);
        appealsTbody.innerHTML = '<tr><td colspan="10" class="text-center text-danger">Failed to load appeals.</td></tr>';
        showNotification('Failed to load appeals.', 'error');
    }
}

// Update appeal decision
async function updateAppealDecision(appealId, decision) {
    try {
        const response = await fetch(`/api/appeals/${appealId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ decision: decision })
        });
        
        if (response.ok) {
            showNotification(`Appeal ${appealId} ${decision.toLowerCase()}`, 'success');
            loadAppealsList(); // Reload the appeals list
        } else {
            throw new Error('Failed to update appeal decision');
        }
    } catch (error) {
        console.error('Error updating appeal decision:', error);
        showNotification('Failed to update appeal decision. Please try again.', 'error');
    }
}

// Load documentation list
async function loadDocumentationList() {
    const docsTbody = document.getElementById('docs-tbody');
    if (!docsTbody) return;
    
    try {
        const response = await fetch('/api/documentation');
        if (!response.ok) throw new Error('Failed to fetch documentation');
        const data = await response.json();
        
        if (data.documentation.length === 0) {
            docsTbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">No documentation found.</td></tr>';
            return;
        }
        
        docsTbody.innerHTML = '';
        data.documentation.forEach(doc => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><strong>${doc.id}</strong></td>
                <td>${doc.requestId}</td>
                <td>${doc.patientId}</td>
                <td>${doc.fileName}</td>
                <td>${doc.uploadTimestamp}</td>
                <td><span class="status-chip status-${doc.status?.toLowerCase().replace(' ', '-') || 'pending'}">${doc.status || 'Pending Review'}</span></td>
                <td>
                    <div class="action-buttons">
                        <button class="action-btn view-doc-btn" onclick="viewDocument('${doc.id}')">
                            <i class="fas fa-eye"></i> View
                        </button>
                        ${doc.status === 'Pending Review' ? `
                            <button class="action-btn approve-doc-btn" onclick="markDocSufficient('${doc.id}')">
                                <i class="fas fa-check"></i> Mark Sufficient
                            </button>
                        ` : ''}
                    </div>
                </td>
            `;
            docsTbody.appendChild(row);
        });
        
    } catch (error) {
        console.error('Error fetching documentation:', error);
        docsTbody.innerHTML = '<tr><td colspan="7" class="text-center text-danger">Failed to load documentation.</td></tr>';
        showNotification('Failed to load documentation.', 'error');
    }
}

// Mark document as sufficient
async function markDocSufficient(docId) {
    try {
        const response = await fetch(`/api/documentation/${docId}/sufficient`, {
            method: 'PUT'
        });
        
        if (response.ok) {
            showNotification(`Document ${docId} marked as sufficient`, 'success');
            loadDocumentationList(); // Reload the documentation list
        } else {
            throw new Error('Failed to mark document as sufficient');
        }
    } catch (error) {
        console.error('Error marking document as sufficient:', error);
        showNotification('Failed to mark document as sufficient. Please try again.', 'error');
    }
}

// Load audit logs
async function loadAuditLogs() {
    const logsTbody = document.getElementById('logs-tbody');
    if (!logsTbody) return;
    
    try {
        const response = await fetch('/audit');
        if (!response.ok) throw new Error('Failed to fetch audit logs');
        const data = await response.json();
        
        if (data.logs.length === 0) {
            logsTbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted">No audit logs found.</td></tr>';
            return;
        }
        
        logsTbody.innerHTML = '';
        data.logs.forEach(log => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${log.timestamp}</td>
                <td><strong>${log.request_id}</strong></td>
                <td>${log.approval_status}</td>
                <td>System</td>
                <td>
                    <strong>Service:</strong> ${log.service_name} (${log.service_type})<br>
                    <strong>Appeal Risk:</strong> ${log.appeal_risk || 'N/A'}<br>
                    <strong>Confidence:</strong> ${log.appeal_confidence || 'N/A'}
                </td>
            `;
            logsTbody.appendChild(row);
        });
        
    } catch (error) {
        console.error('Error fetching audit logs:', error);
        logsTbody.innerHTML = '<tr><td colspan="5" class="text-center text-danger">Failed to load audit logs.</td></tr>';
        showNotification('Failed to load audit logs.', 'error');
    }
}

// Initialize charts
function initializeCharts() {
    // This would initialize charts in a real application
    // For now, we'll just update the dashboard stats
    updateDashboardStats();
}

// Update dashboard stats
function updateDashboardStats() {
    const requests = window.allRequests || [];
    
    // Calculate stats
    const totalRequests = requests.length;
    const approvedRequests = requests.filter(r => r.overallStatus === 'Approved').length;
    const deniedRequests = requests.filter(r => r.overallStatus === 'Denied').length;
    
    // Update stats in the UI - using the correct IDs from the HTML
    const approvedElement = document.querySelector('.stat-card:nth-child(1) .stat-number');
    const deniedElement = document.querySelector('.stat-card:nth-child(2) .stat-number');
    if (approvedElement) approvedElement.textContent = approvedRequests;
    if (deniedElement) deniedElement.textContent = deniedRequests;
}

// View documentation
function viewDocumentation(requestId) {
    // This would open a modal to view documentation in a real application
    showNotification(`View documentation for request ${requestId}`, 'info');
}

// View appeal
function viewAppeal(requestId) {
    // This would open a modal to view appeal details in a real application
    showNotification(`View appeal for request ${requestId}`, 'info');
}

// View document
function viewDocument(docId) {
    // This would open a modal to view the document in a real application
    showNotification(`View document ${docId}`, 'info');
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
            request.requestId.toLowerCase().includes(searchTerm) ||
            request.patientId.toLowerCase().includes(searchTerm) ||
            request.patientName.toLowerCase().includes(searchTerm) ||
            request.diagnosis.toLowerCase().includes(searchTerm) ||
            request.diagnosisCategory.toLowerCase().includes(searchTerm)
        );
    }
    
    // Apply status filter
    if (statusFilter) {
        filteredRequests = filteredRequests.filter(request => 
            request.overallStatus.toLowerCase() === statusFilter.toLowerCase()
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

// Global functions for HTML onclick handlers
window.updateRequestStatusFromAdmin = updateRequestStatusFromAdmin;
window.runRuleEngine = runRuleEngine;
window.viewDocumentation = viewDocumentation;
window.viewAppeal = viewAppeal;
window.updateAppealDecision = updateAppealDecision;
window.viewDocument = viewDocument;
window.markDocSufficient = markDocSufficient;
window.filterRequests = filterRequests;
window.exportRequests = exportRequests;
window.approveAppeal = approveAppeal;
window.denyAppeal = denyAppeal;
window.viewAppealDetails = viewAppealDetails;
window.closeAdminAppealDetailsModal = closeAdminAppealDetailsModal;