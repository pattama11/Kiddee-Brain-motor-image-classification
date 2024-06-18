import { useContext } from 'react';
import Container from 'react-bootstrap/Container';
import Nav from 'react-bootstrap/Nav';
import Navbar from 'react-bootstrap/Navbar';
import logo from '../../public/g8.png';
import signInLogo from '../../public/Frame 179.png';
import UserContext from './UserContext';
import QuestionnaireList from './QuestionnaireList';
import { useLocation } from 'react-router-dom';

function NavScrollExample() {
    const location = useLocation();
    const isSignInPage = location.pathname === '/signin';
    const [user, setUser] = useContext(UserContext);
    
    return (
        <Navbar expand="lg" className="bg-white">
            <Container fluid>
                <Navbar.Brand href="#">
                    <img src={logo} alt="Brand Logo" height="30" className="me-2" /> {/* Added Logo */}
                    Luminos
                </Navbar.Brand>
                <Navbar.Toggle aria-controls="navbarScroll" />
                <Navbar.Collapse id="navbarScroll">
                    <Nav
                        className="ms-auto my-2 my-lg-0"
                        style={{ maxHeight: '100px' }}
                        navbarScroll
                    >
                        <Nav.Link href="/">Home</Nav.Link>
                        <Nav.Link href="#action2">About</Nav.Link>
                        <Nav.Link href="#action2">Services</Nav.Link>
                        <Nav.Link href="#action2">Page</Nav.Link>
                        <Nav.Link href="/">Questionnaires</Nav.Link>
                        <Nav.Link href="#">
                                <img src={signInLogo} alt="Sign In" height="30" />
                        </Nav.Link>
                        {/* {!isSignInPage && (
                        <Nav.Link href="/signin" style={{ visibility: isSignInPage ? 'hidden' : 'visible' }}>
                            Sign in
                        </Nav.Link>
                        )} */}
                        {user ? (
                            <Nav.Link href="/profile" style={{ visibility: isSignInPage ? 'hidden' : 'visible' }}>
                                Welcome, {user.name}!
                            </Nav.Link>
                        ) : (
                            <Nav.Link href="/signin" style={{ visibility: isSignInPage ? 'hidden' : 'visible' }}>
                                Sign in
                            </Nav.Link>
                        )}
                    </Nav>
                </Navbar.Collapse>
            </Container>
        </Navbar>
    );
}

export default NavScrollExample;

