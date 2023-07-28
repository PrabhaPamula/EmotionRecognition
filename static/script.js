var canvasElement = document.getElementById("chart");

var config = {
    type: "bar",
    data: {
        labels: ["Angry","Sad","Neutral","Happy"],
        datasets: [{labels: "Probability Percentage", data: probability}],
    },
};

var chart = new Chart(canvasElement, config);