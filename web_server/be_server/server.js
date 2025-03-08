require("dotenv").config();
const express = require("express");
const cors = require("cors");
const bodyParser = require("body-parser");
const {Pool} = require("pg");
const axios = require("axios");

const app = express();
const PORT = process.env.PORT;

const pool = new Pool({
    user: process.env.DB_USER,
    host: process.env.DB_HOST,
    database: process.env.DB_NAME,
    password: process.env.DB_PASS,
    port: process.env.DB_PORT,
});

app.use(cors());
app.use(bodyParser.json());

app.get("/get_missing_questions", async (req, res) => {
    const max_faiss_id = parseInt(req.query.max_faiss_id, 10) || 0;

    try {
        const result = await pool.query(
            "SELECT id, question FROM kbqa_data WHERE id > $1 ORDER BY id ASC",
            [max_faiss_id]
        );
        const questions = result.rows.map(row => row.question);
        const ids = result.rows.map(row => row.id);
        console.log(`Successfully: ${result}`)
        res.status(200).json({questions, ids});
    } catch (error) {
        console.error("Database Error:", error);
        res.status(500).json({error: "Failed to fetch missing questions"});
    }
});

async function getNextId() {
    const result = await pool.query("SELECT MAX(id) AS max_id FROM kbqa_data");
    return (result.rows[0].max_id || 0) + 1;
}

app.get("/get_answer", async (req, res) => {
    try {
        const {id} = req.query;
        if (!id) {
            return res.status(400).json({error: "Missing ID."});
        }

        const result = await pool.query(
            "SELECT answer FROM kbqa_data WHERE id = $1",
            [id]
        );

        if (result.rows.length === 0) {
            return res.status(404).json({error: "No answer found for the given ID"});
        }
        res.json({answer: result.rows[0].answer});
    } catch (error) {
        console.error("Error fetching answer:", error);
        res.status(500).json({error: "Failed to fetch answer"});
    }
});

app.get("/get_max_id", async (req, res) => {
    try {
        const result = await pool.query("SELECT MAX(id) AS max_id FROM kbqa_data");
        const max_id = result.rows[0].max_id || 0;
        res.json({max_id: max_id});
    } catch (error) {
        console.error("Database error:", error);
        res.status(500).json({error: error.message});
    }
});

app.post("/ask_question", async (req, res) => {
    try {
        var { question } = req.body;
        if (!question) {
            return res.status(400).json({error: "Question is required"});
        }
        question = question.trim().replace(/\s+/g, " ")
        const response = await axios.post(`${process.env.AI_SERVER}/query`, {question});
        const answer = response.data.answer || "No answer available";
        if (answer.trim() === "I don't have a specific answer for this question. Please wait while I forward it to a support staff member.") {
            res.status(200).json({answer});
            return;
        }
        const newId = await getNextId();
        await axios.post(`${process.env.AI_SERVER}/update_faiss`, {id: newId, question});
        await pool.query(
            "INSERT INTO kbqa_data (id, question, answer) VALUES ($1, $2, $3)",
            [newId, question, answer]
        );
        res.status(200).json({answer});
    } catch (error) {
        console.error("Error communicating with AI server:", error.message);
        res.status(500).json({error: "Failed to get response from AI."});
    }
});

app.listen(PORT, () => {
    console.log(`Backend running on http://localhost:${PORT}`);
});