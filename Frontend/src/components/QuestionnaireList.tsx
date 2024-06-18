import { useContext } from "react";
import UserContext from "./UserContext";

import Container from "@mui/material/Container";
import Grid from "@mui/material/Grid";

import FileDownloadOutlinedIcon from "@mui/icons-material/FileDownloadOutlined";

import LogList from "./LogList";
import CardList from "./CardList";

function QuestionnaireList() {
  const [user, setUser] = useContext(UserContext);

  if (true) {
    return (
      <div
        style={{
          backgroundColor: "#FCF8F4",
          minHeight: "100vh",
          paddingTop: "32px",
          paddingBottom: "32px",
        }}
      >
        <Container>
          <h2>Welcome, {user && user.name}!</h2>
          {/* {user && user.role === "teacher" ? ( */}
          {user && user.role == "2" ? (
            <div style={{ paddingTop: "32px" }}>
              <LogList
                img_path="profile.png"
                name="Kiddy Deedee"
                status="Depressed"
                grade="3"
                age="9"
                download_path
              />
            </div>
          ) : (
            <div>
              <p>Here are some questions for students...</p>
              <div className="row" style={{ paddingTop: "32px" }}>
                <CardList
                  img_path="ques-1.png"
                  title="ทดสอบการคบเพื่อน"
                  subtitle="หากคุณอยากรู้ว่าเพื่อนที่คุณคบอยู่เป็นอย่างไร"
                  button_path="/question"
                />
                <CardList
                  img_path="ques-2.png"
                  title="ทดสอบความสุขในการเรียน"
                  subtitle="หากสิ่งที่คุณเรียนอยู่ทำให้คุณมีความสุขจริงๆ"
                  button_path=""
                />
                <CardList
                  img_path="ques-3.png"
                  title="ทดสอบ Mindset"
                  subtitle="หากทัศนคติของคุณ"
                  button_path=""
                />
                <CardList
                  img_path="ques-4.png"
                  title="ทดสอบสภาวะซึมเศร้า"
                  subtitle="หากคุณกำลังมองหาใครสักคนที่เข้าใจคุณอยู่"
                  button_path=""
                />
              </div>
            </div>
          )}
        </Container>
      </div>
    );
  } else {
    return <p>Please sign in to see the questions.</p>;
  }
}

export default QuestionnaireList;
