import { useContext, useState } from "react";
import UserContext from "./UserContext";

import Container from "@mui/material/Container";
import Grid from "@mui/material/Grid";

import Button from "@mui/material/Button";
import LoadingButton from "@mui/lab/LoadingButton";

import FileDownloadOutlinedIcon from "@mui/icons-material/FileDownloadOutlined";

import LogList from "./LogList";
import CardList from "./CardList";
import CircularProgress from "@mui/material/CircularProgress";

import { Link } from "react-router-dom";
import axios from "axios";

function Question() {
  const [state, setState] = useState(false);
  const [loading, setLoading] = useState(false);
  const [question, setQuestion] = useState(0);

  console.log(loading);

  const question_list = [
    "Trouble falling or staying asleep, or sleeping too much",
    "Moving or speaking so slowly that other people could have noticed",
    "Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual",
    "Had nightmares about the event(s) or thought about the event(s) when you did not want to?",
    "Felt numb or detached from people, activities, or your surroundings?",
  ];

  function wait_and_print(delay) {
    // setTimeout(function () {
    //   setQuestion(question + 1);
    //   console.log(question);
    //   setLoading(false);
    //   if (question >= 4) {
    //     setState(true);
    //   }
    // }, delay);
  }

  const handleSubmit = async () => {
    // const response = await fetch(
    //   "http://192.168.181.65:8000/answer?SID=100001&QID=101",
    //   {
    //     method: "GET",
    //     headers: {
    //       "Content-Type": "application/json",
    //     },
    //   }
    // );
    try {
      const response = await axios.get(
        `http://192.168.181.65:8000/answer?SID=100001&QID=101`
      );
      //console.log(response);
      setQuestion(question + 1);
      setLoading(false);
    } catch (error) {
      //console.log("1122", error);
    }
    // if (response.ok) {
    //   setQuestion(question + 1);
    //   setLoading(false);
    //   console.log("1111", loading);
    // }
    if (question >= 4) {
      setState(true);
    }
  };

  return (
    <div
      style={{
        backgroundColor: "#FCF8F4",
        paddingTop: "32px",
        paddingBottom: "32px",
      }}
    >
      <Container>
        {!state ? (
          <Grid
            container
            direction="column"
            justifyContent="center"
            style={{ height: "calc(100vh - 120px)" }}
            gap={3}
          >
            <Grid item>
              <div className="font-super">Question {question + 1}</div>
            </Grid>
            <Grid item>
              <div className="font-header">{question_list[question]}</div>
            </Grid>
            <Grid item>
              <Grid container direction="row" justifyContent="space-between">
                <Grid item>
                  {!loading ? (
                    <Button
                      variant="contained"
                      onClick={async () => {
                        await setLoading(true);
                        //await wait_and_print(3000);
                        handleSubmit();
                      }}
                    >
                      Start
                    </Button>
                  ) : (
                    <CircularProgress />
                  )}
                </Grid>
              </Grid>
            </Grid>
          </Grid>
        ) : (
          <Grid
            container
            direction="column"
            justifyContent="center"
            style={{ height: "calc(100vh - 120px)" }}
            gap={3}
          >
            <Grid item>
              <div className="font-super">Thank You</div>
            </Grid>
            <Grid item>
              <div className="font-header">Your response has been recorded</div>
            </Grid>
            <Grid item>
              [150, 110, 150, 150, 110, 150, 110, 120, 120, 150, 110, 110, 110,
              150, 150, 150, 110, 150, 110, 110, 150, 150, 150, 110, 120, 110,
              150, 120, 110, 120, 110, 110, 150, 120, 120, 110, 150, 110, 120,
              150, 150, 120, 150, 150, 110, 110, 110, 110, 120, 150, 120, 110,
              120, 150, 110, 110, 120, 110, 120, 110]
            </Grid>
            <Grid item>
              <Button component={Link} to="/questionnaires" variant="outlined">
                Back to Questionaire
              </Button>
            </Grid>
          </Grid>
        )}
      </Container>
    </div>
  );
}

export default Question;
