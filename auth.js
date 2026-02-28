// ── Authentication JavaScript ───────────────────────────────

// ── Login Form ───────────────────────────────────────────────
const loginForm = document.getElementById('loginForm');
if (loginForm) {
    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(loginForm);
        const email    = formData.get('email');
        const password = formData.get('password');

        const submitBtn  = loginForm.querySelector('button[type="submit"]');
        const origText   = submitBtn.textContent;
        submitBtn.textContent = 'Signing in…';
        submitBtn.disabled    = true;

        try {
            const res  = await fetch('/api/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password })
            });
            const data = await res.json();

            if (data.success) {
                showNotification('Login successful! Redirecting…', 'success');
                setTimeout(() => window.location.href = data.redirect, 800);
            } else {
                showNotification(data.message || 'Login failed', 'error');
                submitBtn.textContent = origText;
                submitBtn.disabled    = false;
            }
        } catch {
            showNotification('Network error – please try again', 'error');
            submitBtn.textContent = origText;
            submitBtn.disabled    = false;
        }
    });
}

// ── Registration Form ────────────────────────────────────────
const registrationForm = document.getElementById('registrationForm');
if (registrationForm) {
    registrationForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData        = new FormData(registrationForm);
        const password        = formData.get('password');
        const confirmPassword = formData.get('confirmPassword');

        if (password !== confirmPassword) {
            showNotification('Passwords do not match!', 'error');
            return;
        }
        if (password.length < 8) {
            showNotification('Password must be at least 8 characters', 'error');
            return;
        }

        const submitBtn = registrationForm.querySelector('button[type="submit"]');
        const origText  = submitBtn.textContent;
        submitBtn.textContent = 'Creating account…';
        submitBtn.disabled    = true;

        try {
            const res  = await fetch('/api/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    email:   formData.get('email'),
                    password,
                    name:    formData.get('fullname'),
                    company: formData.get('company') || ''
                })
            });
            const data = await res.json();

            if (data.success) {
                showNotification('Account created! Redirecting…', 'success');
                setTimeout(() => window.location.href = data.redirect, 800);
            } else {
                showNotification(data.message || 'Registration failed', 'error');
                submitBtn.textContent = origText;
                submitBtn.disabled    = false;
            }
        } catch {
            showNotification('Network error – please try again', 'error');
            submitBtn.textContent = origText;
            submitBtn.disabled    = false;
        }
    });
}

// ── Social login (placeholder) ───────────────────────────────
document.querySelectorAll('.btn-social').forEach(btn => {
    btn.addEventListener('click', () => {
        const provider = btn.textContent.includes('Google') ? 'Google' : 'GitHub';
        showNotification(`${provider} OAuth would be configured here`, 'info');
    });
});

// ── Notification helper ──────────────────────────────────────
function showNotification(message, type = 'info') {
    document.querySelector('.notification')?.remove();
    const n = document.createElement('div');
    n.className = `notification notification-${type}`;
    n.textContent = message;
    document.body.appendChild(n);
    setTimeout(() => {
        n.style.animation = 'slideOutRight 0.3s ease-out';
        setTimeout(() => n.remove(), 300);
    }, 3000);
}

// ── Password visibility toggle ───────────────────────────────
document.querySelectorAll('input[type="password"]').forEach(input => {
    const wrap = input.parentElement;
    wrap.style.position = 'relative';
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = '👁️';
    btn.style.cssText = `
        position:absolute; right:10px; top:50%;
        transform:translateY(-50%); background:none;
        border:none; cursor:pointer; font-size:1.2rem; opacity:0.6;
    `;
    wrap.appendChild(btn);
    btn.addEventListener('click', () => {
        if (input.type === 'password') {
            input.type = 'text';
            btn.textContent = '🙈';
        } else {
            input.type = 'password';
            btn.textContent = '👁️';
        }
    });
});

console.log('✅ Auth module loaded');
