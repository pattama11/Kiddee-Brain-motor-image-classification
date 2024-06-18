import React, { useContext, useState } from "react";
import { useNavigate } from "react-router-dom";
import UserContext from "./UserContext";

import axios from "axios";
import SHA256 from "crypto-js/sha256";

function SignIn() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState(1);
  const [user, setUser] = useContext(UserContext);
  const navigate = useNavigate();

  const hashedPassword = SHA256(password).toString();

  if (!setUser) {
    throw new Error("setUser is undefined");
  }

  const handleEmailChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setEmail(e.target.value);
    setCheckEmail(e.target.value);
  };

  const handlePasswordChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setPassword(e.target.value);
  };

  const handleRoleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setRole(e.target.value);
    //console.log(role);
  };
  const handleSubmit = async (e) => {
    // navigate("/questionnaires");
    e.preventDefault();
    try {
      const response = await axios.get(
        `http://192.168.181.65:8000/login?username=${email}&password=${hashedPassword.toString()}&role=${role}`
      );
      //console.log("4444", "Login successful", response.data);
      navigate("/questionnaires");
      if (response.data.success) {
        // Navigate to the desired page after successful login
        navigate("/questionnaires");
      } else {
        console.error("Login failed", response.data.error);
        // Display error message or handle failed login
      }
    } catch (error) {
      //console.log("1122", error);
    }
    // setUser({
    //   name: "John",
    //   surname: "Doe",
    //   email,
    //   password,
    //   role,
    // });
    // const response = await fetch(
    //   `http://192.168.181.65:8000/login?username=${email}&password=${password.toString()}`,
    //   {
    //     method: "GET",
    //     headers: {
    //       "Content-Type": "application/json",
    //     },
    //     body: JSON.stringify({
    //       email,
    //       password,
    //       role,
    //     }),
    //   }
    // );

    // if (response.ok) {
    //   const userData = await response.json();
    //   setUser(userData);
    //   console.log("Login successful");
    //   //history.push('/questionnaires'); // navigate to QuestionnaireList page
    //   navigate("/questionnaires");
    // } else {
    //   console.log("Login failed");
    // }
  };

  return (
    <div className="container mt-5">
      <h2 className="text-center">Sign In</h2>
      <form onSubmit={handleSubmit} className="mt-4">
        <div className="form-group">
          <label htmlFor="email">Email:</label>
          <input
            type="email"
            id="email"
            value={email}
            onChange={handleEmailChange}
            required
            className="form-control"
          />
        </div>
        <div className="form-group mt-3">
          <label htmlFor="password">Password:</label>
          <input
            type="password"
            id="password"
            value={password}
            onChange={handlePasswordChange}
            required
            className="form-control"
          />
        </div>
        <div className="form-group mt-3">
          <label htmlFor="role">Role:</label>
          <select
            id="role"
            value={role}
            onChange={handleRoleChange}
            required
            className="form-control"
          >
            <option value="1">Teacher</option>
            <option value="2">Student</option>
          </select>
        </div>
        <button type="submit" className="btn btn-primary mt-3">
          Sign In
        </button>
      </form>
    </div>
  );
}

export default SignIn;
