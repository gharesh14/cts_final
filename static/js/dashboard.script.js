// Global variables
let currentUserRole = '';
let isCreateAccountMode = false;

// DOM elements
const loginModal = document.getElementById('loginModal');
const modalTitle = document.getElementById('modalTitle');
const modalSubtitle = document.getElementById('modalSubtitle');
const loginButtonText = document.getElementById('loginButtonText');
const newUserText = document.getElementById('newUserText');
const createAccountBtn = document.getElementById('createAccountBtn');
const loginForm = document.getElementById('loginForm');

// Show login modal based on user role
function showLoginModal(role) {
    currentUserRole = role;
    isCreateAccountMode = false;
    
    // Update modal content based on role
    if (role === 'doctor') {
        modalTitle.textContent = 'Doctor Login';
        modalSubtitle.textContent = 'Access your healthcare provider dashboard';
        loginButtonText.textContent = 'Login to Doctor Dashboard';
        newUserText.textContent = 'New doctor?';
        createAccountBtn.textContent = 'Create New Account';
        // Show create account option for doctors
        newUserText.style.display = 'block';
        createAccountBtn.style.display = 'inline-block';
    } else if (role === 'admin') {
        modalTitle.textContent = 'Insurance Admin Login';
        modalSubtitle.textContent = 'Access your administrative dashboard';
        loginButtonText.textContent = 'Login to Admin Dashboard';
        newUserText.textContent = 'New admin?';
        createAccountBtn.textContent = 'Create New Account';
        // create account option for admin (only one admin allowed)
        newUserText.style.display = 'block';
        createAccountBtn.style.display = 'inline-block';
    }
    
    // Reset form
    loginForm.reset();
    
    // Show modal
    loginModal.style.display = 'block';
    
    // Add event listeners
    document.addEventListener('click', handleModalClick);
    document.addEventListener('keydown', handleEscapeKey);
}

// Close login modal
function closeLoginModal() {
    loginModal.style.display = 'none';
    isCreateAccountMode = false;
    
    // Remove event listeners
    document.removeEventListener('click', handleModalClick);
    document.removeEventListener('keydown', handleEscapeKey);
}

// Handle modal click outside
function handleModalClick(event) {
    if (event.target === loginModal) {
        closeLoginModal();
    }
}

// Handle escape key
function handleEscapeKey(event) {
    if (event.key === 'Escape') {
        closeLoginModal();
    }
}

// Toggle between login and create account mode
function toggleCreateAccount() {
    isCreateAccountMode = !isCreateAccountMode;
    
    if (isCreateAccountMode) {
        modalTitle.textContent = 'Create New Account';
        modalSubtitle.textContent = 'Set up your credentials for the portal';
        loginButtonText.textContent = 'Create Account';
        createAccountBtn.textContent = 'Back to Login';
        newUserText.style.display = 'none';
    } else {
        if (currentUserRole === 'doctor') {
            modalTitle.textContent = 'Doctor Login';
            modalSubtitle.textContent = 'Access your healthcare provider dashboard';
            loginButtonText.textContent = 'Login to Doctor Dashboard';
            createAccountBtn.textContent = 'Create New Account';
            newUserText.style.display = 'block';
            createAccountBtn.style.display = 'inline-block';
        } else {
            modalTitle.textContent = 'Insurance Admin Login';
            modalSubtitle.textContent = 'Access your administrative dashboard';
            loginButtonText.textContent = 'Login to Admin Dashboard';
            // Keep create account hidden for admin
            newUserText.style.display = 'none';
            createAccountBtn.style.display = 'none';
        }
    }
}

// Handle form submission
loginForm.addEventListener('submit', function(e) {
    e.preventDefault();
    
    const username = document.getElementById('username').value;
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    
    if (!username || !email || !password) {
        showNotification('Please fill in all fields', 'error');
        return;
    }
    
    if (isCreateAccountMode) {
        // Prevent admin account creation
        if (currentUserRole === 'admin') {
            showNotification('Admin account creation is not allowed. Only one admin account exists.', 'error');
            return;
        }
        // Handle account creation for doctors only
        createAccount(username, email, password);
    } else {
        // Handle login
        authenticateUser(username, email, password);
    }
});

// Create new account
async function createAccount(username, email, password) {
    showNotification('Creating account...', 'info');
    
    try {
        const response = await fetch('/create-account', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, email, password, role: currentUserRole })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showNotification(result.message, 'success');
            // Switch back to login mode after successful creation
            setTimeout(() => {
                toggleCreateAccount(); 
            }, 1500);
        } else {
            showNotification(result.message || 'Account creation failed. Please try again.', 'error');
        }
    } catch (error) {
        showNotification('Failed to connect to the server. Please try again later.', 'error');
        console.error('Account creation error:', error);
    }
}

// Authenticate user
async function authenticateUser(username, email, password) {
    showNotification('Authenticating...', 'info');
    
    try {
        const response = await fetch('/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, email, password, role: currentUserRole })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showNotification('Login successful! Redirecting...', 'success');
            // Persist user info for dashboards
            try {
                localStorage.setItem('cts_user', JSON.stringify({ username, email, role: result.role }));
            } catch (e) {}
            setTimeout(() => {
                // Redirect based on the role returned from the backend
                if (result.role === 'doctor') {
                    window.location.href = '/doctor';
                } else if (result.role === 'admin') {
                    window.location.href = '/admin';
                }
            }, 1000);
        } else {
            showNotification(result.message || 'Invalid credentials. Please try again.', 'error');
        }
    } catch (error) {
        showNotification('Failed to connect to the server. Please try again later.', 'error');
        console.error('Login error:', error);
    }
}

// Note: showNotification function is now provided by notification.js
// Add CSS animation for notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    .notification-content {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
    }
    
    .notification-close {
        background: none;
        border: none;
        color: white;
        font-size: 18px;
        cursor: pointer;
        padding: 0;
        line-height: 1;
    }
    
    .notification-close:hover {
        opacity: 0.8;
    }
`;
document.head.appendChild(style);

// Initialize page
document.addEventListener('DOMContentLoaded', function() {
    // Add smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add scroll effect for header
    window.addEventListener('scroll', function() {
        const header = document.querySelector('.header');
        if (window.scrollY > 100) {
            header.style.background = 'rgba(255, 255, 255, 0.98)';
            header.style.boxShadow = '0 2px 20px rgba(0, 0, 0, 0.1)';
        } else {
            header.style.background = 'rgba(255, 255, 255, 0.95)';
            header.style.boxShadow = 'none';
        }
    });
});