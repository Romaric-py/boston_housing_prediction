
document.getElementById('bostonForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const formData = new FormData(event.target);
    const data = Object.fromEntries(formData.entries());

    // Sending the data to a backend endpoint
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
    .then(response => response.json())
    .then(result => {
        alert(`Predicted Price: $${result.price}`);
        
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while predicting the price.');
    });
});