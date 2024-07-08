import { ChatWindow } from "../app/components/ChatWindow";
import { ToastContainer } from "react-toastify";
import { useToken } from "@chakra-ui/react";

export default function App() {
  const title = "Medical Chatbot";
  const [light, dark] = useToken("colors", ["gray.100", "gray.900"]);
  return (
    <>
      <ToastContainer
        position="top-right"
        autoClose={3000}
        hideProgressBar={false}
        newestOnTop={false}
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
      />
      <ChatWindow titleText={title} lightMode={light} darkMode={dark} />
    </>
  );
}
