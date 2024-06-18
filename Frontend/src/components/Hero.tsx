import Container from "react-bootstrap/Container";
import Image from "react-bootstrap/Image";

function Hero() {
  return (
    <section className="hero d-flex flex-column align-items-center">
      <Container className="text-center">
        <h1 className="display-4 mb-3">
          <div>Luminos</div>
        </h1>
        <h1>
          <span>for Success in Education</span>
        </h1>
        <p className="lead mb-4">
          EEG technology helps us understand and nurture student mental health,
          leading to happier, healthier lives.
        </p>
        <p className="lead mb-4">Luminous by Kiddee</p>
        <div className="justify-content-around">
          {/* <Image
            src="https://preview.redd.it/who-would-have-more-appeal-in-real-life-without-the-fame-v0-sv5rx2aps91c1.jpg?width=385&format=pjpg&auto=webp&s=be53049211caab7d323d475a9911e2734e812318"
            style={{ borderRadius: "32px", margin: "10px" }}
            width={195.04}
          />
          <Image
            src="https://preview.redd.it/jordan-barrett-v0-uezbl4ohdlzb1.jpg?width=552&format=pjpg&auto=webp&s=1c425483e7ded2f48eaf1cdbaf69c46d2a8b21d2"
            style={{ borderRadius: "32px", margin: "10px" }}
            width={195.04}
          />
          <Image
            src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQTU6dISvYZngs2M0wDUulbkKAhiLsCF-jsZmyMzjVpbQ&s"
            style={{ borderRadius: "32px", margin: "10px" }}
            width={195.04}
          />
          <Image
            src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS0daaBFwSuBQf9bf7YnfMh7cLQj1-T13L6xpKWquLtqw&s"
            style={{ borderRadius: "32px", margin: "10px" }}
            width={195.04}
          /> */}
          <div></div>
        </div>
      </Container>
      <div
        style={{
          width: "100%",
          height: "100px",
          background: "linear-gradient(90deg, #007D6E, #5EB47C)",
          margin: "10px auto",
        }}
      >
        <Container>
          <div className="gradient-text">
            <div>
              <h1>4000+</h1>
              <h2>Dataset</h2>
            </div>
            <div>
              <h1>28</h1>
              <h2>Team</h2>
            </div>
            <div>
              <h1>2</h1>
              <h2>Domain Expert</h2>
            </div>
          </div>
        </Container>
      </div>
      <Container className="text-center">
        <div className="afterhero">
          <p>Why?</p>
          <h1>
            <span>Nurturing Minds </span>
          </h1>
        </div>
        <Container>
          <div className="afterhero-border">
            <div className="row">
              <div className="col-md-3 border">Box 1</div>
              <div className="col-md-6 border">Box 2</div>
              <div className="col-md-3 border">Box 3</div>
            </div>
          </div>
        </Container>
      </Container>
      <div
        style={{
          width: "100%",
          height: "200px",
          background: "linear-gradient(90deg, #007D6E, #5EB47C)",
          margin: "50px auto",
        }}
      >
        <Container>
          <div className="about">
            <div className="about-content">
              <h1 style={{ color: "white" }}>About us</h1>
              <p>
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do
                eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut
                enim ad minim veniam,{" "}
              </p>
            </div>
          </div>
        </Container>
      </div>

      <Container>
        <h1>Our Technologies</h1>
        <div className="main-container">
          <div className="left-section">
            <div>
              <div className="grid-item">Item 1 (small)</div>
              <div className="grid-item">Item 2 (small)</div>
            </div>
            <div className="grid-item">Item 3 (big)</div>
          </div>

          <div className="right-section">
            <div className="grid-item">Item 4</div>
            <div className="grid-item">Item 5</div>
            <div className="grid-item">Item 6</div>
            <div className="grid-item">Item 7</div>
          </div>
        </div>
        <div className="dark-box">
          <h1>Heading in the Dark Box</h1>
          <p>Some content goes here...</p>
        </div>
      </Container>
    </section>
  );
}

export default Hero;
