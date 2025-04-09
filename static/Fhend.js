function showLoading(show) {
    document.getElementById('loadingMessage').style.display = show ? 'block' : 'none';
    document.getElementById('responseContent').style.display = show ? 'none' : 'block';
}

function askQuestion() {
    const query = document.getElementById('userInput').value.trim();

    if (!query) {
        alert('Please enter a question.');
        return;
    }

    document.getElementById('responseContent').innerHTML = `<p>Loading...</p>`;

    fetch('/ask-question', {
        method: 'POST',  // Correct method
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('responseContent').innerHTML = `<p><strong>GPT Response:</strong> ${data.answer}</p>`;
    })
    .catch(error => {
        document.getElementById('responseContent').innerHTML = `<p>Error: ${error.message}</p>`;
    });
}


function fetchFinancialNews() {
    showLoading(true);

    fetch('/get_news')
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById('responseContent').innerHTML = `<p>Error: ${data.error}</p>`;
        } else {
            let html = `<h3>Latest Financial News:</h3><ul>`;
            data.forEach(article => {
                html += `<li><strong>${article.Headline}</strong> (${article["FinBERT Sentiment"]})</li>`;
            });
            html += '</ul>';
            document.getElementById('responseContent').innerHTML = html;
        }
        showLoading(false);
    })
    .catch(error => {
        document.getElementById('responseContent').innerHTML = `<p>Error: ${error.message}</p>`;
        showLoading(false);
    });
}




