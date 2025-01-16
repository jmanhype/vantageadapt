import { Card, TextInput, Button, Group, Stack } from "@mantine/core";
import { useState } from "react";
import axios from "axios";
import { API_BASE_URL } from "../config";

export default function StrategyForm() {
  const [theme, setTheme] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      await axios.post(`${API_BASE_URL}/api/strategy/generate`, { theme });
      setTheme("");
    } catch (err) {
      console.error("Error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card withBorder>
      <form onSubmit={handleSubmit}>
        <Stack gap="md">
          <TextInput
            label="Strategy Theme"
            value={theme}
            onChange={(e) => setTheme(e.target.value)}
            required
          />
          <Group justify="flex-end">
            <Button type="submit" loading={loading}>
              Generate Strategy
            </Button>
          </Group>
        </Stack>
      </form>
    </Card>
  );
} 