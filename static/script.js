document.addEventListener('DOMContentLoaded', () => {
    const DRAFT_KEY = 'interviewai_setup_draft_v1';

    const jobTitleEl = document.getElementById('job-title');
    const jobTypeEl = document.getElementById('job-type');
    const industryEl = document.getElementById('industry');
    const experienceEl = document.getElementById('experience');
    const jdEl = document.getElementById('jd');
    const languageEl = document.getElementById('language');
    const toneEl = document.getElementById('tone');
    const qCountSlider = document.getElementById('q-count');
    const qCountVal = document.getElementById('q-count-val');
    const customQToggle = document.getElementById('custom-q-toggle');
    const customQArea = document.getElementById('custom-q-area');
    const customQuestionsEl = document.getElementById('custom-questions');
    const showHintsToggle = document.getElementById('show-hints-toggle');
    const skillsWrap = document.getElementById('skills-wrap');
    const skillInput = document.getElementById('skill-input');
    const errorBanner = document.getElementById('error-banner');
    const draftBanner = document.getElementById('draft-banner');
    const discardDraftBtn = document.getElementById('discard-draft-btn');
    const beginBtn = document.getElementById('begin-btn');
    const beginBtnLabel = document.getElementById('begin-btn-label');
    const beginBtnArrow = document.getElementById('begin-btn-arrow');

    let skillsArray = [];

    // ── 1. Slider ──
    if (qCountSlider && qCountVal) {
        qCountSlider.addEventListener('input', (e) => {
            qCountVal.textContent = `${e.target.value} questions`;
        });
    }

    // ── 2. Pill selection (Interview Focus & Difficulty) ──
    const setupPillGroup = (groupId) => {
        const group = document.getElementById(groupId);
        if (!group) return;

        group.addEventListener('click', (e) => {
            if (e.target.classList.contains('pill')) {
                group.querySelectorAll('.pill').forEach(btn => btn.classList.remove('selected'));
                e.target.classList.add('selected');
            }
        });
    };
    setupPillGroup('focus-pills');
    setupPillGroup('diff-pills');

    // ── 3. Custom questions collapse ──
    if (customQToggle && customQArea) {
        customQToggle.addEventListener('change', () => {
            customQArea.classList.toggle('open', customQToggle.checked);
            customQArea.style.display = customQToggle.checked ? 'block' : 'none';
        });
    }

    // ── 4. Skills tag input ──
    function addSkillTag(value) {
        if (!value || skillsArray.includes(value)) return;
        skillsArray.push(value);

        const tag = document.createElement('span');
        tag.className = 'skill-tag';
        tag.innerHTML = `${escapeHtml(value)} <span class="remove-tag" style="cursor:pointer; margin-left:5px;">&times;</span>`;

        tag.querySelector('.remove-tag').addEventListener('click', () => {
            skillsArray = skillsArray.filter(s => s !== value);
            tag.remove();
        });

        skillsWrap.insertBefore(tag, skillInput);
    }

    if (skillInput && skillsWrap) {
        skillInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ',') {
                e.preventDefault();
                const value = skillInput.value.trim().replace(/,$/, '');
                addSkillTag(value);
                skillInput.value = '';
            }
        });
        // Also commit a dangling skill on blur, so it isn't silently lost
        skillInput.addEventListener('blur', () => {
            const value = skillInput.value.trim().replace(/,$/, '');
            if (value) {
                addSkillTag(value);
                skillInput.value = '';
            }
        });
    }

    function escapeHtml(s) {
        return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    // ── 5. Validation ──
    function clearFieldError(el) {
        el.style.borderColor = '';
        el.style.boxShadow = '';
    }
    function markFieldError(el) {
        el.style.borderColor = '#E25454';
        el.style.boxShadow = '0 0 0 3px rgba(226,84,84,0.12)';
    }

    function validate() {
        clearFieldError(jobTitleEl);
        clearFieldError(qCountSlider);

        const errors = [];

        if (!jobTitleEl.value.trim()) {
            errors.push('Job title is required.');
            markFieldError(jobTitleEl);
        }

        const qCount = parseInt(qCountSlider.value, 10);
        if (!Number.isFinite(qCount) || qCount < 3 || qCount > 20) {
            errors.push('Number of questions must be between 3 and 20.');
        }

        if (customQToggle && customQToggle.checked) {
            const customCount = customQuestionsEl.value.split('\n').filter(q => q.trim() !== '').length;
            if (customCount > qCount) {
                errors.push('You have more custom questions than the total question count allows — increase the slider or remove some questions.');
            }
        }

        return errors;
    }

    function showError(message) {
        errorBanner.textContent = message;
        errorBanner.style.display = 'block';
        errorBanner.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
    function hideError() {
        errorBanner.style.display = 'none';
    }

    // ── 6. Draft autosave (local browser storage — this is a real
    // multi-page app, not a Claude.ai artifact, so localStorage is fine here) ──
    function collectDraft() {
        return {
            jobTitle: jobTitleEl.value,
            jobType: jobTypeEl.value,
            industry: industryEl.value,
            experience: experienceEl.value,
            jd: jdEl.value,
            skills: skillsArray,
            focus: document.querySelector('#focus-pills .pill.selected')?.getAttribute('data-val') || 'mixed',
            language: languageEl.value,
            tone: toneEl.value,
            difficulty: document.querySelector('#diff-pills .pill.selected')?.getAttribute('data-val') || 'medium',
            qCount: qCountSlider.value,
            customQuestionsEnabled: customQToggle ? customQToggle.checked : false,
            customQuestions: customQuestionsEl.value,
            showHints: showHintsToggle ? showHintsToggle.checked : false,
        };
    }

    function saveDraft() {
        try {
            localStorage.setItem(DRAFT_KEY, JSON.stringify(collectDraft()));
        } catch (e) {
            // localStorage may be unavailable (private browsing, quota) — fail silently
        }
    }

    function restoreDraft() {
        let raw;
        try {
            raw = localStorage.getItem(DRAFT_KEY);
        } catch (e) {
            return;
        }
        if (!raw) return;

        let draft;
        try {
            draft = JSON.parse(raw);
        } catch (e) {
            return;
        }

        jobTitleEl.value = draft.jobTitle || '';
        jobTypeEl.value = draft.jobType || '';
        industryEl.value = draft.industry || '';
        experienceEl.value = draft.experience || '';
        jdEl.value = draft.jd || '';
        languageEl.value = draft.language || 'en';
        toneEl.value = draft.tone || 'Professional';
        qCountSlider.value = draft.qCount || 8;
        qCountVal.textContent = `${qCountSlider.value} questions`;

        if (Array.isArray(draft.skills)) {
            draft.skills.forEach(addSkillTag);
        }

        if (draft.focus) {
            document.querySelectorAll('#focus-pills .pill').forEach(p => {
                p.classList.toggle('selected', p.getAttribute('data-val') === draft.focus);
            });
        }
        if (draft.difficulty) {
            document.querySelectorAll('#diff-pills .pill').forEach(p => {
                p.classList.toggle('selected', p.getAttribute('data-val') === draft.difficulty);
            });
        }

        if (customQToggle && draft.customQuestionsEnabled) {
            customQToggle.checked = true;
            customQArea.classList.add('open');
            customQArea.style.display = 'block';
        }
        customQuestionsEl.value = draft.customQuestions || '';
        if (showHintsToggle) showHintsToggle.checked = !!draft.showHints;

        if (draftBanner) draftBanner.style.display = 'flex';
    }

    function clearDraft() {
        try {
            localStorage.removeItem(DRAFT_KEY);
        } catch (e) { /* ignore */ }
        if (draftBanner) draftBanner.style.display = 'none';
    }

    if (discardDraftBtn) {
        discardDraftBtn.addEventListener('click', () => {
            clearDraft();
            window.location.reload();
        });
    }

    restoreDraft();

    // Debounced autosave on any change within the form
    let saveTimer = null;
    document.querySelector('main').addEventListener('input', () => {
        clearTimeout(saveTimer);
        saveTimer = setTimeout(saveDraft, 400);
    });
    document.querySelector('main').addEventListener('click', (e) => {
        if (e.target.classList.contains('pill') || e.target.closest('.pill-group')) {
            clearTimeout(saveTimer);
            saveTimer = setTimeout(saveDraft, 400);
        }
    });

    // ── 7. Submit ──
    function setLoading(isLoading) {
        beginBtn.disabled = isLoading;
        beginBtn.style.opacity = isLoading ? '0.7' : '1';
        beginBtn.style.cursor = isLoading ? 'wait' : 'pointer';
        beginBtnLabel.textContent = isLoading ? 'Setting up…' : 'Begin interview';
        beginBtnArrow.textContent = isLoading ? '⏳' : '→';
    }

    async function handleBegin() {
        hideError();
        const errors = validate();
        if (errors.length > 0) {
            showError(errors.join(' '));
            return;
        }

        const payload = {
            jobTitle: jobTitleEl.value.trim(),
            jobType: jobTypeEl.value,
            industry: industryEl.value,
            experience: experienceEl.value,
            jd: jdEl.value,
            skills: skillsArray,
            focus: document.querySelector('#focus-pills .pill.selected')?.getAttribute('data-val') || 'mixed',
            language: languageEl.value,
            tone: toneEl.value,
            difficulty: document.querySelector('#diff-pills .pill.selected')?.getAttribute('data-val') || 'medium',
            qCount: qCountSlider.value,
            customQuestions: customQuestionsEl.value.split('\n').filter(q => q.trim() !== ''),
            showHints: showHintsToggle ? showHintsToggle.checked : false,
        };

        setLoading(true);
        try {
            const response = await fetch('/api/setup', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const result = await response.json();
            if (response.ok && result.status === 'success') {
                clearDraft();
                window.location.href = '/interview';
            } else {
                showError('Error saving setup: ' + (result.message || 'Unknown error.'));
                setLoading(false);
            }
        } catch (err) {
            console.error('Submission failed:', err);
            showError('Could not reach the server. Please check your connection and try again.');
            setLoading(false);
        }
    }

    if (beginBtn) {
        beginBtn.addEventListener('click', handleBegin);
    }

    // ── 8. Save as template (both header + action bar buttons) ──
    function saveAsTemplate() {
        saveDraft();
        const hint = document.getElementById('action-hint');
        if (hint) {
            const original = hint.textContent;
            hint.textContent = 'Saved ✓';
            setTimeout(() => { hint.textContent = original; }, 1800);
        }
    }
    document.getElementById('header-save-btn')?.addEventListener('click', saveAsTemplate);
    document.getElementById('save-template-btn')?.addEventListener('click', saveAsTemplate);
});