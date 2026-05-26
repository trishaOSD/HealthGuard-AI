function predict() {
    let f1 = document.getElementById("f1").value;
    let f2 = document.getElementById("f2").value;
    let f3 = document.getElementById("f3").value;

    fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            features: [f1, f2, f3]
        })
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("result").innerText = data.result;
    });
}