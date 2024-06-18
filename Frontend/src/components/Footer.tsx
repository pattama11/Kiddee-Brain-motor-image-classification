import Container from 'react-bootstrap/Container';

// Define the Footer component
function Footer() {
    return (
        <footer style={{ width: '100%', height: '200px', background: 'linear-gradient(90deg, #007D6E, #5EB47C)',}}> 
            <Container className="text-center d-flex flex-column justify-content-center align-items-center" style={{ height: '100%' }}>
                {/* Your footer content goes here */}
                <p className="text-white">&copy; 2024 Your Company Name</p>
            </Container>
        </footer>
    );
}

export default Footer;
