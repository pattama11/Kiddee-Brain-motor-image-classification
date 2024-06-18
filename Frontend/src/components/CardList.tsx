import { useContext } from "react";
import UserContext from "./UserContext";

import Container from "@mui/material/Container";
import Grid from "@mui/material/Grid";

import { Link } from "react-router-dom";

import FileDownloadOutlinedIcon from "@mui/icons-material/FileDownloadOutlined";

function CardList(props) {
  const { img_path, title, subtitle, button_path } = props;

  return (
    <div className="col-xs-12 col-sm-6 col-md-4 col-lg-3">
      <Link
        to={button_path}
        style={{ textDecoration: "none", color: "#000000" }}
      >
        <div className="card-question" style={{ marginBottom: "16px" }}>
          <img src={img_path} style={{ width: "100%", marginBottom: "32px" }} />
          <div
            className="font-subheader"
            style={{ width: "100%", marginBottom: "8px" }}
          >
            {title}
          </div>
          <div className="font-body">{subtitle}</div>
        </div>
      </Link>
    </div>
  );
}

export default CardList;
