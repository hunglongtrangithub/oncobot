import { toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import { useState } from "react";
import { SourceBubble, Source } from "./SourceBubble";
import {
  VStack,
  Flex,
  Heading,
  HStack,
  Box,
  Button,
  Divider,
  Spacer,
  Spinner,
  CircularProgress,
} from "@chakra-ui/react";
import { apiBaseUrl } from "../utils/constants";
import { InlineCitation } from "./InlineCitation";

export type Message = {
  id: string;
  createdAt?: Date;
  content: string;
  text: string;
  role: "system" | "user" | "assistant" | "function";
  runId?: string;
  sources?: Source[];
  name?: string;
  function_call?: { name: string };
};
export type Feedback = {
  feedback_id: string;
  run_id: string;
  key: string;
  score: number;
  comment?: string;
};

const filterSources = (sources: Source[]) => {
  const filtered: Source[] = [];
  const urlMap = new Map<string, number>();
  const indexMap = new Map<number, number>();
  sources.forEach((source, i) => {
    const { url } = source;
    const index = urlMap.get(url);
    if (index === undefined) {
      urlMap.set(url, i);
      indexMap.set(i, filtered.length);
      filtered.push(source);
    } else {
      const resolvedIndex = indexMap.get(index);
      if (resolvedIndex !== undefined) {
        indexMap.set(i, resolvedIndex);
      }
    }
  });
  return { filtered, indexMap };
};

const createAnswerElements = (
  content: string,
  filteredSources: Source[],
  sourceIndexMap: Map<number, number>,
  highlighedSourceLinkStates: boolean[],
  setHighlightedSourceLinkStates: React.Dispatch<
    React.SetStateAction<boolean[]>
  >,
) => {
  const matches = Array.from(content.matchAll(/\[\^?(\d+)\^?\]/g));
  const elements: JSX.Element[] = [];
  let prevIndex = 0;

  matches.forEach((match) => {
    const sourceNum = parseInt(match[1], 10);
    const resolvedNum = sourceIndexMap.get(sourceNum) ?? 10;
    if (match.index !== null && resolvedNum < filteredSources.length) {
      elements.push(
        <span
          key={`content:${prevIndex}`}
          dangerouslySetInnerHTML={{
            __html: content.slice(prevIndex, match.index),
          }}
        ></span>,
      );
      elements.push(
        <InlineCitation
          key={`citation:${prevIndex}`}
          source={filteredSources[resolvedNum]}
          sourceNumber={resolvedNum}
          highlighted={highlighedSourceLinkStates[resolvedNum]}
          onMouseEnter={() =>
            setHighlightedSourceLinkStates(
              filteredSources.map((_, i) => i === resolvedNum),
            )
          }
          onMouseLeave={() =>
            setHighlightedSourceLinkStates(filteredSources.map(() => false))
          }
        />,
      );
      prevIndex = (match?.index ?? 0) + match[0].length;
    }
  });
  elements.push(
    <span
      key={`content:${prevIndex}`}
      dangerouslySetInnerHTML={{ __html: content.slice(prevIndex) }}
    ></span>,
  );
  return elements;
};

export function ChatMessageBubble(props: {
  message: Message;
  aiEmoji?: string;
  isMostRecent: boolean;
  messageCompleted: boolean;
  selectedChatbot: string;
  conversationId: string;
  lightMode: string;
  darkMode: string;
}) {
  const { role, content, text, runId } = props.message;
  const { conversationId, selectedChatbot, lightMode, darkMode } = props;
  const isUser = role === "user";
  const [isSpeechLoading, setIsSpeechLoading] = useState(false);
  const [isSpeechPlaying, setIsSpeechPlaying] = useState(false);
  const [abortController, setAbortController] =
    useState<AbortController | null>(null);

  const sources = props.message.sources ?? [];
  const { filtered: filteredSources, indexMap: sourceIndexMap } =
    filterSources(sources);

  // Use an array of highlighted states as a state since React
  // complains when creating states in a loop
  const [highlighedSourceLinkStates, setHighlightedSourceLinkStates] = useState(
    filteredSources.map(() => false),
  );
  const answerElements =
    role === "assistant"
      ? createAnswerElements(
          content,
          filteredSources,
          sourceIndexMap,
          highlighedSourceLinkStates,
          setHighlightedSourceLinkStates,
        )
      : [];

  const playMessageAudioWithSpeechSynthesis = (
    message: string,
    controller: AbortController,
  ) => {
    console.log("play message audio with speech synthesis");

    // Check if speech synthesis is supported
    if (!window.speechSynthesis) {
      console.error("Speech synthesis not supported in this browser.");
      toast.error("Speech synthesis not supported in this browser.");
      return;
    }

    // Create a new SpeechSynthesisUtterance instance
    const utterance = new SpeechSynthesisUtterance(message);

    // Set the pitch and rate
    utterance.pitch = 1;
    utterance.rate = 1;

    // Play the speech
    window.speechSynthesis.speak(utterance);

    console.log("playing audio");

    utterance.onstart = () => {
      console.log("audio playing");
      setIsSpeechLoading(false);
      setIsSpeechPlaying(true);
    };

    utterance.onend = () => {
      console.log("audio ended");
      setIsSpeechPlaying(false);
    };

    // Handle abort controller's signal for speech synthesis
    controller.signal.addEventListener("abort", () => {
      window.speechSynthesis.cancel();
      console.log("Speech synthesis was canceled");
      toast.info("Speech synthesis was canceled.");
      setIsSpeechLoading(false);
      setIsSpeechPlaying(false);
    });
  };

  const playMessageAudio = async (message: string, selectedChatbot: string) => {
    console.log("play message audio");
    if (selectedChatbot === "") {
      // TODO: may have to change "voice chat" to something else later
      toast.error("Please select a chatbot in voice chat to play audio.");
      return;
    }

    const formData = new FormData();
    const botAudioBlob = await fetch(`/bots/${selectedChatbot}.mp3`).then(
      (res) => res.blob(),
    );
    formData.append("bot_voice_file", botAudioBlob, `${selectedChatbot}.mp3`);
    formData.append("message", message);
    formData.append("conversationId", conversationId);
    formData.append("chatbot", selectedChatbot);

    const controller = new AbortController();
    setAbortController(controller);

    setIsSpeechLoading(true);
    try {
      const audioResponse = await fetch(apiBaseUrl + "/text_to_speech", {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });
      if (!audioResponse.ok) {
        const errorMessage = await audioResponse.text();
        console.error("Failed to fetch audio:", errorMessage);
        playMessageAudioWithSpeechSynthesis(message, controller);
        setIsSpeechLoading(false);
        return;
      }

      const audioBlob = await audioResponse.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);

      console.log("playing audio");
      audio.play();

      audio.onplay = () => {
        console.log("audio playing");
        setIsSpeechLoading(false);
        setIsSpeechPlaying(true);
      };

      audio.onended = () => {
        console.log("audio ended");
        setIsSpeechPlaying(false);
      };

      // Handle abort controller's signal for audio playback
      controller.signal.addEventListener("abort", () => {
        if (audio) {
          audio.pause();
          audio.currentTime = 0; // Reset the audio playback position
        }
        console.log("Audio playback was paused");
        toast.info("Audio playback was canceled.");
        setIsSpeechLoading(false);
        setIsSpeechPlaying(false);
      });
    } catch (error: any) {
      if (error.name === "AbortError") {
        console.log("Audio playback was aborted");
        toast.info("Audio playback was canceled.");
      } else {
        console.error("Audio playback error:", error);
        toast.error("Failed to play audio.");
      }
      setIsSpeechLoading(false);
    }
  };

  const cancelOperation = () => {
    if (abortController) {
      abortController.abort();
      setIsSpeechLoading(false);
      setIsSpeechPlaying(false);
      setAbortController(null);
    }
  };

  return (
    <VStack align="start" spacing={5} pb={5}>
      {!isUser && filteredSources.length > 0 && (
        <>
          <Flex direction={"column"} width={"100%"}>
            <VStack spacing={"5px"} align={"start"} width={"100%"}>
              <Heading
                fontSize="lg"
                fontWeight={"medium"}
                mb={1}
                paddingBottom={"10px"}
              >
                Sources
              </Heading>
              <HStack spacing={"10px"} maxWidth={"100%"} overflow={"auto"}>
                {filteredSources.map((source, index) => (
                  <Box key={index} alignSelf={"stretch"} width={40}>
                    <SourceBubble
                      lightMode={lightMode}
                      darkMode={darkMode}
                      source={source}
                      highlighted={highlighedSourceLinkStates[index]}
                      onMouseEnter={() =>
                        setHighlightedSourceLinkStates(
                          filteredSources.map((_, i) => i === index),
                        )
                      }
                      onMouseLeave={() =>
                        setHighlightedSourceLinkStates(
                          filteredSources.map(() => false),
                        )
                      }
                      runId={runId}
                    />
                  </Box>
                ))}
              </HStack>
            </VStack>
          </Flex>

          <Heading size="lg" fontWeight="medium">
            Answer
          </Heading>
        </>
      )}

      {isUser ? (
        <Heading size="lg" fontWeight="medium">
          {content}
        </Heading>
      ) : (
        <Box className="whitespace-pre-wrap">{answerElements}</Box>
      )}

      {props.message.role !== "user" &&
        // props.isMostRecent &&
        props.messageCompleted && (
          <HStack spacing={2}>
            <Button
              size="sm"
              variant="outline"
              colorScheme="blue"
              onClick={(e) => {
                e.preventDefault();
                if (isSpeechLoading || isSpeechPlaying) {
                  cancelOperation();
                  return;
                }
                playMessageAudio(text, selectedChatbot);
              }}
            >
              🔉 Audio
            </Button>
            <Spacer />
            {isSpeechPlaying ? (
              <Spinner />
            ) : isSpeechLoading ? (
              <CircularProgress isIndeterminate size="30px" />
            ) : null}
          </HStack>
        )}
      {!isUser && <Divider mt={4} mb={4} />}
    </VStack>
  );
}
