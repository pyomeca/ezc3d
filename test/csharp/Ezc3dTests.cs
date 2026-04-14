using System;
using Xunit;
using Ezc3d;

public class Ezc3dTests
{
    [Fact]
    public void CanLoadFile_AndAccessBasicData()
    {
        // Arrange
        var path = "test/c3dTestFiles/Vicon.c3d";

        // Act
        var file = new c3d(path);

        // Assert - Header
        Assert.True(file.header().nb3dPoints() > 0);

        // Assert - Parameters
        var used = file.parameters()
                       .group("POINT")
                       .parameter("USED")
                       .valuesAsInt();

        Assert.True(used.Count > 0);
        Assert.True(used[0] > 0);

        // Assert - Data
        var point = file.data()
                        .frame(0)
                        .points()
                        .point(0);

        Assert.False(double.IsNaN(point.x()));
    }
}