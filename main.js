
        // Add event listeners for buttons
        document.querySelectorAll('.flipkart-btn').forEach(button => {
            button.addEventListener('click', function() {
                const productName = this.closest('.product-details').querySelector('.product-title').textContent;
                alert(`Redirecting to Flipkart to buy ${productName}`);
            });
        });
        
        document.querySelectorAll('.amazon-btn').forEach(button => {
            button.addEventListener('click', function() {
                const productName = this.closest('.product-details').querySelector('.product-title').textContent;
                alert(`Redirecting to Amazon to buy ${productName}`);
            });
        });
        
        document.querySelectorAll('.compare-btn').forEach(button => {
            button.addEventListener('click', function() {
                const productName = this.closest('.product-details').querySelector('.product-title').textContent;
                alert(`Compare prices for ${productName}`);
            });
        });
        
        // Smooth scrolling for navigation links
        document.querySelectorAll('.nav-links a').forEach(anchor => {
            anchor.addEventListener('click', function(e) {
                e.preventDefault();
                const targetId = this.getAttribute('href');
                const targetElement = document.querySelector(targetId);
                
                window.scrollTo({
                    top: targetElement.offsetTop - 100,
                    behavior: 'smooth'
                });
            });
        });

        document.getElementById("productForm").addEventListener("submit", function (e) {
            e.preventDefault(); // Prevent page refresh
            const userInput = document.getElementById("productInput").value.trim();
            if (userInput) {
                document.getElementById("resultText").innerText = `Thank You. We will consider your suggestion.`;
                // You can send this input to backend or use it for filtering
                // Example: redirect to search results page
                // window.location.href = `/search.html?query=${encodeURIComponent(userInput)}`;
            }
        });

      