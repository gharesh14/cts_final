/**
 * Global Notification System
 * Provides consistent notification functionality across all pages
 */

// Global notification function that works on any page
function showNotification(message, type = 'info') {
    // Remove existing notifications to avoid clutter
    const existingNotifications = document.querySelectorAll('.global-notification');
    existingNotifications.forEach(notification => notification.remove());
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `global-notification global-notification-${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <div class="notification-icon">
                ${getNotificationIcon(type)}
            </div>
            <div class="notification-message">${message}</div>
            <button class="notification-close" onclick="this.parentElement.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;
    
    // Add styles
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        left: 50%;
        transform: translateX(-50%);
        background: ${getNotificationColor(type)};
        color: white;
        padding: 16px 24px;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        z-index: 10000;
        max-width: 500px;
        min-width: 300px;
        animation: slideDown 0.4s ease-out;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 14px;
        font-weight: 500;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.style.animation = 'slideUp 0.3s ease-in';
            setTimeout(() => {
                if (notification.parentElement) {
                    notification.remove();
                }
            }, 300);
        }
    }, 5000);
}

// Get appropriate icon for notification type
function getNotificationIcon(type) {
    switch (type) {
        case 'success':
            return '<i class="fas fa-check-circle"></i>';
        case 'error':
            return '<i class="fas fa-exclamation-circle"></i>';
        case 'warning':
            return '<i class="fas fa-exclamation-triangle"></i>';
        case 'info':
        default:
            return '<i class="fas fa-info-circle"></i>';
    }
}

// Get appropriate color for notification type
function getNotificationColor(type) {
    switch (type) {
        case 'success':
            return 'linear-gradient(135deg, #10B981, #059669)';
        case 'error':
            return 'linear-gradient(135deg, #EF4444, #DC2626)';
        case 'warning':
            return 'linear-gradient(135deg, #F59E0B, #D97706)';
        case 'info':
        default:
            return 'linear-gradient(135deg, #3B82F6, #2563EB)';
    }
}

// Add CSS animations for notifications
function addNotificationStyles() {
    if (document.getElementById('notification-styles')) return;
    
    const style = document.createElement('style');
    style.id = 'notification-styles';
    style.textContent = `
        @keyframes slideDown {
            from {
                opacity: 0;
                transform: translateX(-50%) translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(-50%) translateY(0);
            }
        }
        
        @keyframes slideUp {
            from {
                opacity: 1;
                transform: translateX(-50%) translateY(0);
            }
            to {
                opacity: 0;
                transform: translateX(-50%) translateY(-20px);
            }
        }
        
        .global-notification .notification-content {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .global-notification .notification-icon {
            font-size: 18px;
            flex-shrink: 0;
        }
        
        .global-notification .notification-message {
            flex: 1;
            line-height: 1.4;
        }
        
        .global-notification .notification-close {
            background: none;
            border: none;
            color: white;
            cursor: pointer;
            padding: 4px;
            border-radius: 4px;
            transition: background-color 0.2s;
            flex-shrink: 0;
        }
        
        .global-notification .notification-close:hover {
            background-color: rgba(255,255,255,0.2);
        }
        
        .global-notification .notification-close i {
            font-size: 14px;
        }
    `;
    
    document.head.appendChild(style);
}

// Initialize notification styles when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', addNotificationStyles);
} else {
    addNotificationStyles();
}

// Enhanced submit button handler that automatically shows notifications
function handleSubmitWithNotification(formElement, submitFunction) {
    if (!formElement) return;
    
    formElement.addEventListener('submit', async function(event) {
        event.preventDefault();
        
        const submitButton = formElement.querySelector('button[type="submit"], .submit-btn');
        const originalText = submitButton ? submitButton.innerHTML : '';
        const originalDisabled = submitButton ? submitButton.disabled : false;
        
        // Show loading state
        if (submitButton) {
            submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Submitting...';
            submitButton.disabled = true;
        }
        
        try {
            // Call the provided submit function
            const result = await submitFunction(formElement);
            
            // Show success notification
            showNotification(result.message || 'Submission successful!', 'success');
            
            // Reset form if specified
            if (result.resetForm !== false) {
                formElement.reset();
            }
            
        } catch (error) {
            console.error('Submission error:', error);
            showNotification(error.message || 'Submission failed. Please try again.', 'error');
        } finally {
            // Restore button state
            if (submitButton) {
                submitButton.innerHTML = originalText;
                submitButton.disabled = originalDisabled;
            }
        }
    });
}

// Auto-detect and enhance all submit buttons on the page
function enhanceAllSubmitButtons() {
    // Find all forms with submit buttons
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        const submitButton = form.querySelector('button[type="submit"], .submit-btn');
        if (submitButton && !form.hasAttribute('data-notification-enhanced')) {
            form.setAttribute('data-notification-enhanced', 'true');
            
            // Add a default handler if no custom one exists
            if (!form.hasAttribute('data-custom-submit')) {
                handleSubmitWithNotification(form, async (formElement) => {
                    // Default form submission logic
                    const formData = new FormData(formElement);
                    const data = Object.fromEntries(formData.entries());
                    
                    // Try to find an action URL
                    const action = formElement.action || window.location.pathname;
                    const method = formElement.method || 'POST';
                    
                    const response = await fetch(action, {
                        method: method,
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(data)
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    
                    return await response.json();
                });
            }
        }
    });
    
    // Find standalone submit buttons
    const standaloneSubmitButtons = document.querySelectorAll('button[type="submit"]:not(form button)');
    standaloneSubmitButtons.forEach(button => {
        if (!button.hasAttribute('data-notification-enhanced')) {
            button.setAttribute('data-notification-enhanced', 'true');
            
            button.addEventListener('click', async function(event) {
                event.preventDefault();
                
                const originalText = button.innerHTML;
                const originalDisabled = button.disabled;
                
                // Show loading state
                button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Submitting...';
                button.disabled = true;
                
                try {
                    // Try to find associated form or use default behavior
                    const form = button.closest('form');
                    if (form) {
                        form.dispatchEvent(new Event('submit'));
                    } else {
                        // Default success for standalone buttons
                        showNotification('Action completed successfully!', 'success');
                    }
                } catch (error) {
                    console.error('Button click error:', error);
                    showNotification('Action failed. Please try again.', 'error');
                } finally {
                    // Restore button state
                    button.innerHTML = originalText;
                    button.disabled = originalDisabled;
                }
            });
        }
    });
}

// Initialize when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', enhanceAllSubmitButtons);
} else {
    enhanceAllSubmitButtons();
}

// Export for use in other scripts
window.showNotification = showNotification;
window.handleSubmitWithNotification = handleSubmitWithNotification;
window.enhanceAllSubmitButtons = enhanceAllSubmitButtons;
