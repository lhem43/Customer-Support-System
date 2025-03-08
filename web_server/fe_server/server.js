const express = require("express")
const path = require("path")

const app = express()
const PORT = 2222;

app.use(express.static(path.join(__dirname)))

app.get("/", (req, res) => {
    res.sendFile(path.join(__dirname, "index.html"))
})

app.listen(PORT, () => {
    console.log(`Website is running at http://localhost:${PORT}`)
})