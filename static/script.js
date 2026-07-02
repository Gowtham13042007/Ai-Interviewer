document.addEventListener('DOMContentLoaded', () => {
    // 1. Slider Functionality
    const qCountSlider = document.getElementById('q-count');
    const qCountVal = document.getElementById('q-count-val');

    if (qCountSlider && qCountVal) {
        qCountSlider.addEventListener('input', (e) => {
            qCountVal.textContent = `${e.target.value} questions`;
        });
    }

    // 2. Pill Selection Behavior (Interview Focus & Difficulty)
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

    const toggle = document.getElementById('custom-q-toggle');
    const collapseArea = document.getElementById('custom-q-area');

    if (toggle && collapseArea) {
        toggle.addEventListener('change', () => {
            if (toggle.checked) {
                collapseArea.classList.add('show'); // Ensure your CSS handles .collapse.show display properties
                collapseArea.style.display = "block"; 
            } else {
                collapseArea.classList.remove('show');
                collapseArea.style.display = "none";
            }
        });
    }

    // 4. Skills Tag Input Management
    const skillsWrap = document.getElementById('skills-wrap');
    const skillInput = document.getElementById('skill-input');
    let skillsArray = [];

    if (skillInput && skillsWrap) {
        skillInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ',') {
                e.preventDefault();
                const value = skillInput.value.trim().replace(/,$/, '');
                
                if (value && !skillsArray.includes(value)) {
                    skillsArray.push(value);
                    
                    // Create visual tag element
                    const tag = document.createElement('span');
                    tag.className = 'skill-tag'; // Style this in your CSS
                    tag.innerHTML = `${value} <span class="remove-tag" style="cursor:pointer; margin-left:5px;">&times;</span>`;
                    
                    // Wire up remove button
                    tag.querySelector('.remove-tag').addEventListener('click', () => {
                        skillsArray = skillsArray.filter(s => s !== value);
                        tag.remove();
                    });
                    
                    skillsWrap.insertBefore(tag, skillInput);
                }
                skillInput.value = '';
            }
        });
    }

  
    const beginBtn = document.querySelector('.action-buttons .btn-primary');

    if (beginBtn) {
        beginBtn.addEventListener('click', async (e) => {
            e.preventDefault();


            const payload = {
                jobTitle: document.getElementById('job-title').value,
                jobType: document.getElementById('job-type').value,
                industry: document.getElementById('industry').value,
                experience: document.getElementById('experience').value,
                jd: document.getElementById('jd').value,
                skills: skillsArray,
                focus: document.querySelector('#focus-pills .pill.selected')?.getAttribute('data-val') || 'mixed',
                language: document.getElementById('language').value,
                tone: document.getElementById('tone').value,
                difficulty: document.querySelector('#diff-pills .pill.selected')?.getAttribute('data-val') || 'medium',
                qCount: qCountSlider ? qCountSlider.value : 8,
                customQuestions: document.getElementById('custom-questions').value.split('\n').filter(q => q.trim() !== ''),
                showHints: document.querySelector('.toggle-row:last-of-type input[type="checkbox"]')?.checked || false
            };

            
            try {
                const response = await fetch('/api/setup', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const result = await response.json();
                if (response.ok && result.status === 'success') {
                   
                    window.location.href = '/interview';
                } else {
                    alert('Error saving setup configs: ' + result.message);
                }
            } catch (err) {
                console.error('Submission failed:', err);
                alert('Server connectivity issue.');
            }
        });
    }
});
