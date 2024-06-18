import React, { useContext, useState } from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import App from "./App";
import { UserProvider } from "./components/UserContext";
import SignIn from "./components/SignIn";
import Navbar from "./components/Navbar";
import QuestionnaireList from "./components/QuestionnaireList";
import Question from "./components/question";

const rootElement = document.getElementById("root");

if (rootElement) {
  const root = ReactDOM.createRoot(rootElement);
  root.render(
    <React.StrictMode>
      <BrowserRouter>
        <UserProvider>
          <Navbar />
          <Routes>
            <Route path="/" element={<App />} />
            <Route path="/signin" element={<SignIn />} />
            <Route path="/questionnaires" element={<QuestionnaireList />} />
            <Route path="/question" element={<Question />} />
          </Routes>
        </UserProvider>
      </BrowserRouter>
    </React.StrictMode>
  );
}
