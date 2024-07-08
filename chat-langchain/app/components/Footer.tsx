import { FaGithub } from "react-icons/fa";

export default function Footer(props: { colorMode: string }) {
  return (
    <footer className="flex justify-center relative py-4">
      <a
        href="https://github.com/hunglongtrangithub/chat-langchain"
        target="_blank"
        className="flex items-center"
      >
        <FaGithub size={20} />
        <span>View Source</span>
      </a>
    </footer>
  );
}
