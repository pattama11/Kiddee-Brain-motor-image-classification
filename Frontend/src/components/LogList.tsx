import { useContext } from "react";
import UserContext from "./UserContext";

import Container from "@mui/material/Container";
import Grid from "@mui/material/Grid";

import FileDownloadOutlinedIcon from "@mui/icons-material/FileDownloadOutlined";

function LogList(props) {
  const { img_path, name, status, grade, age, download_path } = props;

  return (
    <Grid item xs={12}>
      <div
        className="card-question d-flex flex-row gap-3"
        style={{
          marginBottom: "16px",
        }}
      >
        <Grid container gap={3}>
          <Grid item>
            <img src={img_path} style={{ width: "48px", height: "48px" }} />
          </Grid>
          <Grid item xs>
            <Grid container direction="column">
              <Grid item>
                <Grid container direction="row" alignItems="center" gap={1}>
                  <Grid item>
                    <div className="font-subheader">{name}</div>
                  </Grid>
                  <Grid item>
                    <div className="tag">{status}</div>
                  </Grid>
                </Grid>
              </Grid>
              <Grid item>
                <div className="font-body">Grade {grade}</div>
              </Grid>
              <Grid item>
                <div className="font-body">
                  {age} {age > 1 ? "years old" : "year old"}
                </div>
              </Grid>
            </Grid>
          </Grid>
          <Grid item>
            <FileDownloadOutlinedIcon />
          </Grid>
        </Grid>
      </div>
    </Grid>
  );
}

export default LogList;
